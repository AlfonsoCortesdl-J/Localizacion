#!/usr/bin/env python3
"""
visualize_simulation.py — Reproduce visualmente el vuelo simulado.

Lee lo que ya generó el pipeline (data/*.csv, data/camera_ground/*.png, mapa de referencia)
y muestra en una ventana:
  - Vista cenital con el ortomapa, trayectoria real (Ground Truth) y las estimadas
    (VO pura, VO+TRN, INS puro, EKF) con su error en vivo.
  - Fotograma de la camara nadir sincronizado con el instante de vuelo.
  - Horizonte artificial + telemetria.
  - Grafica de error vs tiempo con cursor.

Uso (desde la raiz del proyecto, con el .venv activado):
    python src/visualize_simulation.py                 # ventana interactiva
    python src/visualize_simulation.py --speed 2       # el doble de rapido
    python src/visualize_simulation.py --save plots/vuelo.mp4   # exporta video (sin ventana)
    python src/visualize_simulation.py --inspect       # lista las columnas de cada CSV

Teclas: ESPACIO pausa · + / - velocidad · R reiniciar · Q o ESC salir

Si alguna trayectoria estimada no aparece, ejecuta --inspect y rellena OVERRIDES abajo.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Si la deteccion automatica de columnas falla, indica aqui (archivo, columna_este, columna_norte)
# en metros relativos al punto de inicio. Ejemplo:
#   OVERRIDES = {"vo": ("visual_navigation_results.csv", "vo_east_m", "vo_north_m")}
OVERRIDES: dict = {}

W, H = 1280, 720
MAP_W = 720
COLORS = {  # BGR
    "gt": (255, 255, 255),
    "vo": (0, 165, 255),
    "trn": (80, 220, 80),
    "ins": (60, 60, 255),
    "ekf": (255, 255, 0),
}
LABELS = {
    "gt": "Ground Truth",
    "vo": "VO pura",
    "trn": "VO + TRN",
    "ins": "INS puro",
    "ekf": "EKF (INS+vision)",
}

# -------------------------------------------------------------------- utilidades de columnas


def find_col(df, cands, exclude=()):
    low = {c: c.lower() for c in df.columns}
    for cand in cands:
        for c, l in low.items():
            if l == cand:
                return c
    for cand in cands:
        for c, l in low.items():
            if cand in l and not any(e in l for e in exclude):
                return c
    return None


def time_vector(df, n_default, duration):
    col = find_col(df, ["sim_time_sec", "sim_time_s", "time_s", "time_sec", "time", "t_s", "t"],
                   exclude=("stamp",))
    if col is not None:
        t = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        t = t - t[0]
        if np.all(np.isfinite(t)) and np.all(np.diff(t) >= 0) and t[-1] > 0:
            return t
    return np.linspace(0, duration, len(df)) if n_default else np.arange(len(df), dtype=float)


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, prefix + str(k).lower() + "."))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{prefix}{i}."))
    else:
        out[prefix.rstrip(".")] = d
    return out


# -------------------------------------------------------------------- carga de datos


def read_map_center(meta_path):
    """(lat, lon, ancho_m) del mapa segun reference_map_meta.json, o None."""
    if not meta_path.exists():
        return None
    flat = {k: v for k, v in flatten(json.load(open(meta_path, encoding="utf-8"))).items()
            if isinstance(v, (int, float))}
    clat = next((v for k, v in flat.items() if "center" in k and "lat" in k), None)
    clon = next((v for k, v in flat.items() if "center" in k and "lon" in k), None)
    width = next((v for k, v in flat.items() if k.endswith("width_m")), None)
    if clat is None or clon is None:
        return None
    return float(clat), float(clon), (float(width) if width else None)


class Flight:
    """Ground Truth en coordenadas locales (E, N) respecto al primer punto."""

    def __init__(self, df, origin=None):
        self.df = df
        c = lambda *a, **k: find_col(df, *a, **k)
        self.col = {
            "lat": c(["lat_geod_deg", "latitude_deg", "lat"]),
            "lon": c(["lon_geod_deg", "longitude_deg", "lon"]),
            "alt": c(["altitude_msl_m", "altitude", "alt"]),
            "roll": c(["roll_deg", "roll"]),
            "pitch": c(["pitch_deg", "pitch"]),
            "yaw": c(["yaw_deg", "heading_deg", "yaw", "heading"]),
            "tas": c(["airspeed_tas_mps", "groundspeed_mps"]),
            "tas_kts": c(["airspeed_tas_kts"]),
        }
        missing = [k for k in ("lat", "lon") if self.col[k] is None]
        if missing:
            sys.exit(f"flight_log.csv no tiene columnas de {missing}. Columnas: {list(df.columns)}")
        self.t = time_vector(df, False, 60.0)
        if origin is not None:      # mismo origen y factores que sensor_fusion_ekf.py
            self.lat0, self.lon0 = origin
            self.kN = 111_320.0
        else:
            self.lat0 = float(df[self.col["lat"]].iloc[0])
            self.lon0 = float(df[self.col["lon"]].iloc[0])
            self.kN = 111_132.0
        self.kE = 111_320.0 * np.cos(np.radians(self.lat0))
        self.E = (df[self.col["lon"]].to_numpy(float) - self.lon0) * self.kE
        self.N = (df[self.col["lat"]].to_numpy(float) - self.lat0) * self.kN
        self.n = len(df)

    def val(self, key, i, default=0.0):
        c = self.col.get(key)
        if c is None and key == "tas" and self.col.get("tas_kts"):
            return float(self.df[self.col["tas_kts"]].iloc[i]) * 0.514444
        return float(self.df[c].iloc[i]) if c is not None else default


EXCL_COMMON = ("mps", "err", "inlier", "gt", "true", "truth", "ref", "cov", "sigma", "std", "var",
               "vel", "att", "euler", "roll", "pitch", "yaw", "head", "rumbo", "alt", "down",
               "time", "stamp", "match", "score", "corr_ok", "flag")
GROUP_FILE = {
    "vo": "visual_navigation_results.csv", "trn": "visual_navigation_results.csv",
    "ins": "fused_navigation_log.csv", "ekf": "fused_navigation_log.csv",
}
GROUPS = {
    "vo": dict(inc=("vo",), exc=("trn", "map", "fus", "ekf", "ins")),
    "trn": dict(inc=("trn", "map", "fus", "corr"), exc=("ekf", "ins")),
    "ins": dict(inc=("ins",), exc=("ekf",)),
    "ekf": dict(inc=("ekf", "fus", "fused"), exc=("ins",)),
}


def to_xy(df, a, b, fl):
    """Convierte el par de columnas (a, b) en (E, N) metros; reconoce lat/lon en grados."""
    va = pd.to_numeric(df[a], errors="coerce").to_numpy(float)
    vb = pd.to_numeric(df[b], errors="coerce").to_numpy(float)
    la, lb = a.lower(), b.lower()
    if "lat" in la and "lon" in lb:
        return (vb - fl.lon0) * fl.kE, (va - fl.lat0) * fl.kN
    if "lon" in la and "lat" in lb:
        return (va - fl.lon0) * fl.kE, (vb - fl.lat0) * fl.kN
    return va, vb


def match_track(df, group, fl, tdur):
    """Busca el par de columnas del grupo cuya trayectoria mejor coincide con el GT."""
    g = GROUPS[group]
    cols = [c for c in df.columns
            if any(k in c.lower() for k in g["inc"])
            and not any(k in c.lower() for k in g["exc"] + EXCL_COMMON)
            and pd.api.types.is_numeric_dtype(df[c])]
    t = time_vector(df, True, tdur)
    best = None
    print(f"  [{group}] columnas candidatas: {cols if cols else 'ninguna'}")
    for a, b in itertools.permutations(cols, 2):
        E, N = to_xy(df, a, b, fl)
        if not (np.all(np.isfinite(E)) and np.all(np.isfinite(N))):
            continue
        Ei, Ni = np.interp(fl.t, t, E), np.interp(fl.t, t, N)
        d = float(np.median(np.hypot(Ei - fl.E, Ni - fl.N)))
        if best is None or d < best[0]:
            best = (d, a, b, Ei, Ni)
    if best is not None:
        print(f"  [{group}] mejor par: {best[1]}, {best[2]}  (distancia mediana al GT: {best[0]:.0f} m)")
    if best is not None and best[0] < 1200:   # descarta pares sin relacion con el GT
        return best[1], best[2], best[3], best[4]
    return None


def load_tracks(data, fl):
    tdur = float(fl.t[-1])
    files = {}
    for name in ("visual_navigation_results.csv", "fused_navigation_log.csv"):
        p = data / name
        if p.exists():
            files[name] = pd.read_csv(p)
    tracks = {}
    for group in GROUPS:
        if group in OVERRIDES:
            fname, ca, cb = OVERRIDES[group]
            df = files.get(fname) if fname in files else pd.read_csv(data / fname)
            E, N = to_xy(df, ca, cb, fl)
            t = time_vector(df, True, tdur)
            tracks[group] = (f"{fname}:{ca},{cb}", np.interp(fl.t, t, E), np.interp(fl.t, t, N))
            continue
        for fname, df in files.items():
            if fname != GROUP_FILE[group]:
                continue
            m = match_track(df, group, fl, tdur)
            if m:
                tracks[group] = (f"{fname}:{m[0]},{m[1]}", m[2], m[3])
                break
    return tracks


def load_map(data, fl):
    """Devuelve (imagen, E_min, E_max, N_min, N_max) o None."""
    img_path, meta_path = data / "reference_map.png", data / "reference_map_meta.json"
    if not img_path.exists():
        return None
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    ctr = read_map_center(meta_path)
    if ctr is not None and ctr[2]:
        half = ctr[2] / 2.0
        ce, cn = (ctr[1] - fl.lon0) * fl.kE, (ctr[0] - fl.lat0) * fl.kN
        return img, ce - half, ce + half, cn - half, cn + half
    lat_min = lat_max = lon_min = lon_max = None
    if meta_path.exists():
        flat = {k: v for k, v in flatten(json.load(open(meta_path, encoding="utf-8"))).items()
                if isinstance(v, (int, float))}
        for k, v in flat.items():
            if "lat" in k and ("min" in k or "south" in k):
                lat_min = v
            elif "lat" in k and ("max" in k or "north" in k):
                lat_max = v
            elif "lon" in k and ("min" in k or "west" in k):
                lon_min = v
            elif "lon" in k and ("max" in k or "east" in k):
                lon_max = v
        if None in (lat_min, lat_max, lon_min, lon_max):
            clat = next((v for k, v in flat.items() if "center" in k and "lat" in k), None)
            clon = next((v for k, v in flat.items() if "center" in k and "lon" in k), None)
            gsd = next((v for k, v in flat.items() if "gsd" in k), None)
            if None not in (clat, clon, gsd):
                hh, ww = img.shape[:2]
                dlat = gsd * hh / 2 / fl.kN
                dlon = gsd * ww / 2 / fl.kE
                lat_min, lat_max, lon_min, lon_max = clat - dlat, clat + dlat, clon - dlon, clon + dlon
    if None in (lat_min, lat_max, lon_min, lon_max):
        print("[aviso] No pude leer los limites geograficos del mapa; se omite el fondo.")
        return None
    return (img,
            (lon_min - fl.lon0) * fl.kE, (lon_max - fl.lon0) * fl.kE,
            (lat_min - fl.lat0) * fl.kN, (lat_max - fl.lat0) * fl.kN)


# -------------------------------------------------------------------- dibujo


class View:
    def __init__(self, fl, tracks, mp):
        xs = [fl.E] + [t[1] for t in tracks.values()]
        ys = [fl.N] + [t[2] for t in tracks.values()]
        xs, ys = np.concatenate(xs), np.concatenate(ys)
        e0, e1, n0, n1 = xs.min(), xs.max(), ys.min(), ys.max()
        span_e, span_n = max(e1 - e0, 600.0), max(n1 - n0, 600.0)
        self.s = min((MAP_W - 40) / (span_e * 1.15), (H - 40) / (span_n * 1.15))
        self.ec, self.nc = (e0 + e1) / 2, (n0 + n1) / 2
        self.bg = self._background(mp)

    def px(self, E, N):
        x = (np.asarray(E) - self.ec) * self.s + MAP_W / 2
        y = H / 2 - (np.asarray(N) - self.nc) * self.s
        return np.stack([x, y], axis=-1)

    def _background(self, mp):
        bg = np.full((H, MAP_W, 3), 28, np.uint8)
        if mp is not None:
            img, e0, e1, n0, n1 = mp
            hh, ww = img.shape[:2]
            ge, gn = (e1 - e0) / ww, (n1 - n0) / hh
            tx = (e0 - self.ec) * self.s + MAP_W / 2
            ty = H / 2 - (n1 - self.nc) * self.s
            M = np.float32([[ge * self.s, 0, tx], [0, gn * self.s, ty]])
            warp = cv2.warpAffine(img, M, (MAP_W, H), flags=cv2.INTER_AREA,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(28, 28, 28))
            bg = (warp * 0.75).astype(np.uint8)
        # cuadricula
        span_m = MAP_W / self.s
        step = next((s for s in (50, 100, 200, 250, 500, 1000, 2000) if span_m / s <= 8), 2000)
        e_lo = self.ec - MAP_W / 2 / self.s
        n_lo = self.nc - H / 2 / self.s
        for e in np.arange(np.ceil(e_lo / step) * step, e_lo + span_m, step):
            x = int((e - self.ec) * self.s + MAP_W / 2)
            cv2.line(bg, (x, 0), (x, H), (70, 70, 70), 1)
        for n in np.arange(np.ceil(n_lo / step) * step, n_lo + H / self.s, step):
            y = int(H / 2 - (n - self.nc) * self.s)
            cv2.line(bg, (0, y), (MAP_W, y), (70, 70, 70), 1)
        # barra de escala
        L = int(step * self.s)
        cv2.line(bg, (20, H - 25), (20 + L, H - 25), (255, 255, 255), 2)
        cv2.putText(bg, f"{step} m", (20, H - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return bg


def draw_aircraft(img, x, y, yaw_deg, size=14):
    a = np.radians(yaw_deg)
    fwd = np.array([np.sin(a), -np.cos(a)])      # pantalla: norte = arriba
    right = np.array([np.cos(a), np.sin(a)])
    p = np.array([x, y])
    pts = np.array([p + fwd * size, p - fwd * size * 0.7 + right * size * 0.6,
                    p - fwd * size * 0.3, p - fwd * size * 0.7 - right * size * 0.6], np.int32)
    cv2.fillConvexPoly(img, pts, (255, 255, 255))
    cv2.polylines(img, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)


def horizon(roll, pitch, R=70):
    S = 4 * R
    img = np.zeros((S, S, 3), np.uint8)
    cy = S // 2
    ppd = R / 25.0
    hy = int(np.clip(cy + pitch * ppd, 0, S - 1))
    img[:hy] = (225, 160, 60)
    img[hy:] = (45, 95, 140)
    cv2.line(img, (0, hy), (S, hy), (255, 255, 255), 2)
    for d in (-20, -10, 10, 20):
        y = int(hy - d * ppd)
        cv2.line(img, (cy - R // 4, y), (cy + R // 4, y), (255, 255, 255), 1)
    M = cv2.getRotationMatrix2D((S / 2, S / 2), roll, 1.0)
    img = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR)
    img = img[cy - R:cy + R, cy - R:cy + R].copy()
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(mask, (R, R), R - 2, 255, -1)
    img[mask == 0] = (20, 20, 20)
    cv2.circle(img, (R, R), R - 2, (200, 200, 200), 2)
    cv2.line(img, (R - 30, R), (R - 10, R), (0, 255, 255), 3)
    cv2.line(img, (R + 10, R), (R + 30, R), (0, 255, 255), 3)
    cv2.circle(img, (R, R), 3, (0, 255, 255), -1)
    return img


def text(img, s, x, y, scale=0.5, color=(230, 230, 230), thick=1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class Renderer:
    def __init__(self, data, fl, tracks, mp):
        self.fl, self.tracks = fl, tracks
        self.view = View(fl, tracks, mp)
        self.frames = sorted((data / "camera_ground").glob("*.png"))
        self.T = float(fl.t[-1])
        self.err = {k: np.hypot(v[1] - fl.E, v[2] - fl.N) for k, v in tracks.items()}
        self.px = {"gt": self.view.px(fl.E, fl.N)}
        for k, v in tracks.items():
            self.px[k] = self.view.px(v[1], v[2])
        self.chart = self._chart_base()
        self._cache = (None, None)

    # ---- grafica de error
    CX, CY, CW, CH = 740, 580, 520, 125

    def _chart_base(self):
        img = np.full((self.CH, self.CW, 3), 24, np.uint8)
        cv2.rectangle(img, (0, 0), (self.CW - 1, self.CH - 1), (90, 90, 90), 1)
        self.ymax = max([e.max() for e in self.err.values()] + [10.0]) * 1.1
        for k, e in self.err.items():
            xs = 6 + self.fl.t / self.T * (self.CW - 12)
            ys = self.CH - 6 - e / self.ymax * (self.CH - 22)
            cv2.polylines(img, [np.stack([xs, ys], 1).astype(np.int32)], False, COLORS[k], 1, cv2.LINE_AA)
        text(img, f"error de posicion (max {self.ymax:.0f} m)", 8, 14, 0.4)
        return img

    def camera(self, t):
        if not self.frames:
            img = np.full((405, 540, 3), 30, np.uint8)
            text(img, "sin fotogramas en data/camera_ground", 60, 200)
            return img
        i = min(int(t / self.T * len(self.frames)), len(self.frames) - 1)
        if self._cache[0] != i:
            im = cv2.imread(str(self.frames[i]))
            if im is None:
                im = np.zeros((405, 540, 3), np.uint8)
            self._cache = (i, cv2.resize(im, (540, 405)))
        img = self._cache[1].copy()
        text(img, f"camara nadir  frame {i + 1}/{len(self.frames)}", 8, 18, 0.5, (0, 255, 255))
        return img

    def render(self, t):
        fl = self.fl
        i = int(np.clip(np.searchsorted(fl.t, t), 0, fl.n - 1))
        canvas = np.full((H, W, 3), 18, np.uint8)

        # --- mapa
        m = self.view.bg.copy()
        cv2.polylines(m, [self.px["gt"].astype(np.int32)], False, (110, 110, 110), 1, cv2.LINE_AA)
        for k in ("vo", "ins", "trn", "ekf", "gt"):
            if k in self.px and i > 1:
                cv2.polylines(m, [self.px[k][:i + 1].astype(np.int32)], False, COLORS[k],
                              2 if k != "gt" else 3, cv2.LINE_AA)
            if k in self.px and k != "gt":
                x, y = self.px[k][i]
                cv2.circle(m, (int(x), int(y)), 5, COLORS[k], -1, cv2.LINE_AA)
        gx, gy = self.px["gt"][i]
        draw_aircraft(m, gx, gy, fl.val("yaw", i))
        y0 = 22
        for k in ("gt", "vo", "trn", "ins", "ekf"):
            if k != "gt" and k not in self.tracks:
                continue
            cv2.line(m, (12, y0 - 4), (34, y0 - 4), COLORS[k], 3)
            lbl = LABELS[k] + ("" if k == "gt" else f"  {self.err[k][i]:6.1f} m")
            text(m, lbl, 42, y0, 0.5, (255, 255, 255))
            y0 += 20
        canvas[:, :MAP_W] = m

        # --- camara
        canvas[10:415, 730:1270] = self.camera(t)

        # --- horizonte + telemetria
        hz = horizon(fl.val("roll", i), fl.val("pitch", i))
        canvas[430:570, 740:880] = hz
        lines = [
            f"t        {fl.t[i]:6.1f} / {self.T:.0f} s",
            f"altitud  {fl.val('alt', i):7.1f} m",
            f"TAS      {fl.val('tas', i):7.1f} m/s",
            f"roll     {fl.val('roll', i):+7.1f} deg",
            f"pitch    {fl.val('pitch', i):+7.1f} deg",
            f"rumbo    {fl.val('yaw', i) % 360:7.1f} deg",
        ]
        for j, s in enumerate(lines):
            text(canvas, s, 900, 452 + j * 20, 0.52, (240, 240, 240))
        text(canvas, f"{self.speed_label}", 1130, 452, 0.5, (0, 200, 255))

        # --- grafica
        ch = self.chart.copy()
        cx = int(6 + t / self.T * (self.CW - 12))
        cv2.line(ch, (cx, 0), (cx, self.CH), (255, 255, 255), 1)
        canvas[self.CY:self.CY + self.CH, self.CX:self.CX + self.CW] = ch
        return canvas

    speed_label = ""


# -------------------------------------------------------------------- principal


def inspect(data):
    for p in sorted(data.glob("*.csv")):
        try:
            df = pd.read_csv(p, nrows=3)
            n = sum(1 for _ in open(p, "rb")) - 1
            print(f"\n{p.name}  ({n} filas)\n  " + ", ".join(df.columns))
        except Exception as e:
            print(f"{p.name}: {e}")
    meta = data / "reference_map_meta.json"
    if meta.exists():
        print("\nreference_map_meta.json:\n" + meta.read_text(encoding="utf-8"))


def main():
    here = Path(__file__).resolve().parent
    root_default = here.parent if here.name == "src" else here
    ap = argparse.ArgumentParser(description="Reproduce visualmente la simulacion de vuelo")
    ap.add_argument("--root", type=Path, default=root_default, help="raiz del proyecto")
    ap.add_argument("--speed", type=float, default=1.0, help="velocidad de reproduccion")
    ap.add_argument("--save", type=Path, help="exportar a video .mp4 (modo sin ventana)")
    ap.add_argument("--inspect", action="store_true", help="listar columnas de los CSV")
    ap.add_argument("--loop", action="store_true", help="repetir al terminar")
    args = ap.parse_args()

    data = args.root / "data"
    if args.inspect:
        inspect(data)
        return
    if not (data / "flight_log.csv").exists():
        sys.exit(f"No encuentro {data / 'flight_log.csv'}. Corre antes el pipeline (paso 1).")

    ctr = read_map_center(data / "reference_map_meta.json")
    fl = Flight(pd.read_csv(data / "flight_log.csv"), origin=(ctr[0], ctr[1]) if ctr else None)
    print("Origen de coordenadas:", "centro del mapa" if ctr else "primer punto del vuelo")
    tracks = load_tracks(data, fl)
    mp = load_map(data, fl)
    print("Trayectorias detectadas:")
    for k in GROUPS:
        print(f"  {LABELS[k]:18s}", tracks[k][0] if k in tracks else "NO ENCONTRADA (usa --inspect y OVERRIDES)")
    print("Mapa de fondo:", "si" if mp else "no")

    r = Renderer(data, fl, tracks, mp)
    T = r.T

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        n = int(T / (args.speed / 30.0))
        r.speed_label = f"x{args.speed:g}"
        for f in range(n):
            vw.write(r.render(min(f * args.speed / 30.0, T)))
        vw.release()
        print(f"Video guardado en {args.save}  ({n} cuadros)")
        return

    win = "Simulacion de vuelo — Cessna 172P"
    try:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    except cv2.error:
        sys.exit("Tu OpenCV no tiene interfaz grafica (opencv-python-headless?). Usa --save vuelo.mp4")
    speed, paused, t = args.speed, False, 0.0
    last = time.perf_counter()
    while True:
        now = time.perf_counter()
        if not paused:
            t += (now - last) * speed
        last = now
        if t >= T:
            if args.loop:
                t = 0.0
            else:
                t, paused = T, True
        r.speed_label = f"x{speed:g}" + ("  PAUSA" if paused else "")
        cv2.imshow(win, r.render(t))
        k = cv2.waitKey(15) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord(" "):
            paused = not paused
            if t >= T:
                t, paused = 0.0, False
        elif k in (ord("+"), ord("=")):
            speed = min(speed * 1.5, 16)
        elif k in (ord("-"), ord("_")):
            speed = max(speed / 1.5, 0.1)
        elif k in (ord("r"), ord("R")):
            t, paused = 0.0, False
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
