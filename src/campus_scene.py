"""
campus_scene.py
===============
Escena georreferenciada del TecMM Campus Zapopan y generador de la misión de
inspección del dron.

- Proyección local ENU (x = Este, z = -Norte) usada por el visor Three.js.
- Huellas reales de edificios, canchas, vialidades y áreas verdes obtenidas de
  OpenStreetMap (Overpass API). Si no hay red, cae a la detección satelital
  existente en ``data/campus_buildings_detected.json``.
- Alturas estimadas por nombre/tipo de edificio (OSM no publica niveles para
  este campus); quedan documentadas en el JSON generado.
- Misión de inspección con trayectoria suave, corredor sin GPS y dinámica de
  error modelada para GT / INS / EKF / VIO.

Uso:
    python src/campus_scene.py                 # descarga/actualiza y genera todo
    python src/campus_scene.py --offline       # solo datos locales
    python src/campus_scene.py --mission-only  # regenera la misión
"""

from __future__ import annotations

import argparse
import json
import math
import random
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
META_PATH = DATA_DIR / "tecmm_campus_meta.json"
SCENE_PATH = DATA_DIR / "campus_scene.json"
MISSION_PATH = DATA_DIR / "campus_mission.json"
DETECTED_PATH = DATA_DIR / "campus_buildings_detected.json"

OVERPASS_MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)

# Alturas estimadas (m). OSM del campus no incluye building:levels.
HEIGHT_BY_NAME = {
    "posgrado": 13.0,
    "sistemas": 15.0,
    "electronica": 15.0,
    "electromecanica": 15.0,
    "basicas": 11.0,
    "industrial": 13.0,
    "organo": 11.0,
    "redi": 13.0,
    "domo": 13.0,
    "nuevo": 15.0,
}
DEFAULT_HEIGHT_M = 9.0

# Anchos de vialidad (m) usados para dibujar cintas en el visor.
ROAD_WIDTH_M = {
    "primary": 14.0,
    "secondary": 11.0,
    "tertiary": 9.0,
    "residential": 7.0,
    "service": 4.5,
    "track": 3.5,
    "footway": 2.0,
    "path": 2.0,
}

# Puntos de interés para etiquetas del visor (x, z en metros locales).
POIS = [
    {"name": "Acceso principal", "x": -120.0, "z": 405.0, "kind": "gate"},
    {"name": "Edificio de Posgrado", "x": -150.0, "z": -40.0, "kind": "academic"},
    {"name": "Ing. en Sistemas", "x": 30.0, "z": -60.0, "kind": "academic"},
    {"name": "Ing. Electrónica", "x": 120.0, "z": -40.0, "kind": "academic"},
    {"name": "Ing. Electromecánica", "x": 120.0, "z": 60.0, "kind": "academic"},
    {"name": "Ciencias Básicas", "x": -30.0, "z": 120.0, "kind": "academic"},
    {"name": "Domo Universitario", "x": 260.0, "z": -180.0, "kind": "sport"},
    {"name": "REDi Zapopan", "x": 300.0, "z": 160.0, "kind": "innovation"},
    {"name": "Cancha", "x": 250.0, "z": 250.0, "kind": "sport"},
]


# ---------------------------------------------------------------------------
# Proyección geodésica local
# ---------------------------------------------------------------------------
class CampusProjection:
    """Proyección equirectangular local centrada en el campus."""

    def __init__(self, meta: Dict[str, Any]):
        self.meta = meta
        self.center_lat = float(meta["center_lat"])
        self.center_lon = float(meta["center_lon"])
        self.k_lat = float(meta["k_lat"])
        self.k_lon = float(meta["k_lon"])
        self.width_m = float(meta["width_m"])
        self.height_m = float(meta["height_m"])

    def to_local(self, lat: float, lon: float) -> Tuple[float, float]:
        x = (lon - self.center_lon) * self.k_lon
        z = -(lat - self.center_lat) * self.k_lat
        return round(x, 2), round(z, 2)

    def to_geodetic(self, x: float, z: float) -> Tuple[float, float]:
        lon = self.center_lon + x / self.k_lon
        lat = self.center_lat - z / self.k_lat
        return round(lat, 7), round(lon, 7)


def load_meta(path: Path = META_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Descarga OSM
# ---------------------------------------------------------------------------
def fetch_osm_elements(meta: Dict[str, Any], timeout: int = 60) -> List[Dict[str, Any]]:
    """Descarga geometrías de edificios, vialidades y áreas verdes del campus."""
    pad = 0.0008
    bbox = (
        meta["lat_min"] - pad,
        meta["lon_min"] - pad,
        meta["lat_max"] + pad,
        meta["lon_max"] + pad,
    )
    b = ",".join(f"{v:.7f}" for v in bbox)
    query = f"""[out:json][timeout:{timeout}];
(
  way["building"]({b});
  way["leisure"~"pitch|track|sports_centre|park|garden|playground"]({b});
  way["amenity"~"parking|school|university|college|place_of_worship"]({b});
  way["landuse"~"grass|forest|meadow|residential|education|commercial"]({b});
  way["highway"~"footway|service|residential|tertiary|secondary|primary|path|track"]({b});
  way["natural"~"wood|scrub|water"]({b});
  way["waterway"]({b});
);
out geom tags;"""
    payload = urllib.parse.urlencode({"data": query}).encode("utf-8")
    last_error: Optional[Exception] = None
    for mirror in OVERPASS_MIRRORS:
        try:
            req = urllib.request.Request(
                mirror, data=payload, headers={"User-Agent": "TecMM-Campus-3D/2.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            elements = data.get("elements", [])
            if elements:
                print(f"[campus] OSM: {len(elements)} elementos vía {mirror}")
                return elements
        except Exception as exc:  # pragma: no cover - depende de la red
            last_error = exc
            print(f"[campus] OSM falló en {mirror}: {exc}")
    if last_error:
        print(f"[campus] Sin conexión OSM ({last_error}); se usará el respaldo local.")
    return []


# ---------------------------------------------------------------------------
# Conversión de elementos OSM a escena
# ---------------------------------------------------------------------------
def _polygon_area(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, z1 = points[i]
        x2, z2 = points[(i + 1) % len(points)]
        area += x1 * z2 - x2 * z1
    return abs(area) * 0.5


def _centroid(points: Sequence[Sequence[float]]) -> Tuple[float, float]:
    if not points:
        return 0.0, 0.0
    return (
        round(sum(p[0] for p in points) / len(points), 2),
        round(sum(p[1] for p in points) / len(points), 2),
    )


def _estimate_height(tags: Dict[str, str]) -> Tuple[float, str]:
    """Altura en metros y su procedencia (dato OSM o estimación)."""
    raw_height = tags.get("height")
    if raw_height:
        try:
            return float(str(raw_height).replace("m", "").strip()), "osm:height"
        except ValueError:
            pass
    levels = tags.get("building:levels")
    if levels:
        try:
            return float(levels) * 3.2, "osm:levels"
        except ValueError:
            pass
    name = (tags.get("name") or "").lower()
    for key, height in HEIGHT_BY_NAME.items():
        if key in name:
            return height, "estimada:tipología"
    building_type = tags.get("building", "")
    if building_type in {"roof", "garage", "garages", "shed", "hut"}:
        return 4.0, "estimada:tipo"
    return DEFAULT_HEIGHT_M, "estimada:promedio"


def build_scene_from_osm(
    meta: Dict[str, Any], elements: Iterable[Dict[str, Any]]
) -> Dict[str, Any]:
    projection = CampusProjection(meta)
    buildings: List[Dict[str, Any]] = []
    pitches: List[Dict[str, Any]] = []
    parking: List[Dict[str, Any]] = []
    greens: List[Dict[str, Any]] = []
    water: List[Dict[str, Any]] = []
    roads: List[Dict[str, Any]] = []

    for element in elements:
        tags = element.get("tags", {}) or {}
        geometry = element.get("geometry") or []
        if len(geometry) < 2:
            continue
        points = [list(projection.to_local(p["lat"], p["lon"])) for p in geometry]
        closed = len(points) >= 4 and points[0] == points[-1]
        if closed:
            points = points[:-1]

        if "building" in tags and len(points) >= 3:
            height, source = _estimate_height(tags)
            centroid = _centroid(points)
            buildings.append(
                {
                    "id": f"osm-{element.get('id')}",
                    "name": tags.get("name") or "Edificio",
                    "type": tags.get("building") or "yes",
                    "height_m": round(height, 1),
                    "height_source": source,
                    "levels": tags.get("building:levels"),
                    "polygon": points,
                    "centroid": list(centroid),
                    "area_m2": round(_polygon_area(points), 1),
                }
            )
        elif tags.get("leisure") in {"pitch", "track", "sports_centre", "playground"} and len(points) >= 3:
            pitches.append(
                {
                    "name": tags.get("name") or tags.get("leisure"),
                    "kind": tags.get("leisure"),
                    "sport": tags.get("sport"),
                    "polygon": points,
                    "centroid": list(_centroid(points)),
                }
            )
        elif tags.get("amenity") == "parking" and len(points) >= 3:
            parking.append({"polygon": points, "centroid": list(_centroid(points))})
        elif ("landuse" in tags or "natural" in tags) and len(points) >= 3:
            if tags.get("natural") == "water" or tags.get("waterway"):
                water.append({"polygon": points})
            else:
                greens.append({"polygon": points, "kind": tags.get("landuse") or tags.get("natural")})
        elif "highway" in tags and len(points) >= 2:
            road_type = tags["highway"]
            roads.append(
                {
                    "type": road_type,
                    "name": tags.get("name"),
                    "width_m": ROAD_WIDTH_M.get(road_type, 3.0),
                    "points": points,
                }
            )

    return {
        "meta": {
            "name": meta.get("institution", "TecMM Campus Zapopan"),
            "alias": meta.get("alias"),
            "center_lat": meta["center_lat"],
            "center_lon": meta["center_lon"],
            "width_m": meta["width_m"],
            "height_m": meta["height_m"],
            "image": "/data/tecmm_campus_map.png",
            "image_px": [1536, 1536],
            "projection": "ENU local: x=Este, z=-Norte",
        },
        "buildings": sorted(buildings, key=lambda b: -b["area_m2"]),
        "pitches": pitches,
        "parking": parking,
        "greens": greens,
        "water": water,
        "roads": roads,
        "pois": POIS,
        "source": "OpenStreetMap + alturas estimadas",
    }


def build_scene_fallback(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Respaldo: cajas a partir de los blobs detectados en la ortofoto."""
    projection = CampusProjection(meta)
    buildings: List[Dict[str, Any]] = []
    if DETECTED_PATH.exists():
        with open(DETECTED_PATH, "r", encoding="utf-8") as f:
            detected = json.load(f)
        for item in detected:
            x = float(item["x3d"])
            z = float(item["z3d"])
            w = float(item["w_m"]) / 2.0
            h = float(item["h_m"]) / 2.0
            # La detección entrega cajas alineadas al eje de la imagen.
            polygon = [[x - w, z - h], [x + w, z - h], [x + w, z + h], [x - w, z + h]]
            buildings.append(
                {
                    "id": f"det-{item['id']}",
                    "name": "Edificio detectado",
                    "type": "yes",
                    "height_m": 8.0,
                    "height_source": "estimada:respaldo",
                    "levels": None,
                    "polygon": [[round(px, 2), round(pz, 2)] for px, pz in polygon],
                    "centroid": [round(x, 2), round(z, 2)],
                    "area_m2": round(float(item["area_m2"]), 1),
                }
            )
    return {
        "meta": {
            "name": meta.get("institution", "TecMM Campus Zapopan"),
            "alias": meta.get("alias"),
            "center_lat": meta["center_lat"],
            "center_lon": meta["center_lon"],
            "width_m": meta["width_m"],
            "height_m": meta["height_m"],
            "image": "/data/tecmm_campus_map.png",
            "image_px": [1536, 1536],
            "projection": "ENU local: x=Este, z=-Norte",
        },
        "buildings": buildings,
        "pitches": [],
        "parking": [],
        "greens": [],
        "water": [],
        "roads": [],
        "pois": POIS,
        "source": "detección satelital (respaldo)",
    }


def _point_in_polygon(x: float, z: float, polygon: Sequence[Sequence[float]]) -> bool:
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, z1 = polygon[i]
        x2, z2 = polygon[(i + 1) % n]
        if (z1 > z) != (z2 > z):
            xin = (x2 - x1) * (z - z1) / (z2 - z1 + 1e-12) + x1
            if x < xin:
                inside = not inside
    return inside


def add_procedural_trees(scene: Dict[str, Any], seed: int = 7) -> None:
    """Árboles decorativos deterministas que evitan edificios y canchas."""
    rng = random.Random(seed)
    meta = scene["meta"]
    half_w = meta["width_m"] / 2.0
    half_h = meta["height_m"] / 2.0
    obstacles = [b["polygon"] for b in scene["buildings"]]
    obstacles += [p["polygon"] for p in scene.get("pitches", [])]
    obstacles += [p["polygon"] for p in scene.get("parking", [])]

    trees: List[List[float]] = []
    attempts = 0
    while len(trees) < 220 and attempts < 6000:
        attempts += 1
        x = rng.uniform(-half_w * 0.95, half_w * 0.95)
        z = rng.uniform(-half_h * 0.95, half_h * 0.95)
        # Evitar el centro de pista y edificios.
        if any(_point_in_polygon(x, z, poly) for poly in obstacles):
            continue
        # Los árboles se concentran en perímetro y franjas verdes.
        edge = max(abs(x) / half_w, abs(z) / half_h)
        if edge < 0.45 and rng.random() > 0.25:
            continue
        if any((x - tx) ** 2 + (z - tz) ** 2 < 14.0 ** 2 for tx, tz, *_ in trees):
            continue
        radius = rng.uniform(2.0, 4.6)
        height = rng.uniform(5.0, 11.0)
        trees.append([round(x, 1), round(z, 1), round(radius, 1), round(height, 1)])
    scene["trees"] = trees


def ensure_scene(force: bool = False, offline: bool = False) -> Dict[str, Any]:
    if SCENE_PATH.exists() and not force:
        with open(SCENE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    meta = load_meta()
    elements = [] if offline else fetch_osm_elements(meta)
    scene = build_scene_from_osm(meta, elements) if elements else build_scene_fallback(meta)
    add_procedural_trees(scene)
    SCENE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCENE_PATH, "w", encoding="utf-8") as f:
        json.dump(scene, f, ensure_ascii=False, indent=1)
    print(
        f"[campus] Escena guardada en {SCENE_PATH} "
        f"({len(scene['buildings'])} edificios, {len(scene['roads'])} vialidades)"
    )
    return scene


# ---------------------------------------------------------------------------
# Misión de inspección
# ---------------------------------------------------------------------------
def _catmull_rom(points: Sequence[Sequence[float]], samples_per_segment: int = 24) -> List[List[float]]:
    """Interpola una polilínea con Catmull-Rom (sin duplicar extremos)."""
    pts = [list(p) for p in points]
    if len(pts) < 2:
        return pts
    padded = [pts[0]] + pts + [pts[-1]]
    out: List[List[float]] = []
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = padded[i], padded[i + 1], padded[i + 2], padded[i + 3]
        for s in range(samples_per_segment):
            u = s / samples_per_segment
            u2, u3 = u * u, u * u * u
            x = 0.5 * (
                2 * p1[0]
                + (-p0[0] + p2[0]) * u
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * u2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * u3
            )
            z = 0.5 * (
                2 * p1[1]
                + (-p0[1] + p2[1]) * u
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * u2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * u3
            )
            out.append([x, z])
    out.append(pts[-1])
    return out


def _resample_path(path: Sequence[Sequence[float]], speed_mps: float, hz: float = 10.0) -> List[Dict[str, float]]:
    """Recorre la polilínea a velocidad constante y devuelve muestras temporales."""
    step = speed_mps / hz
    samples: List[Dict[str, float]] = []
    t = 0.0
    for i in range(len(path) - 1):
        x0, z0 = path[i]
        x1, z1 = path[i + 1]
        seg = math.hypot(x1 - x0, z1 - z0)
        if seg < 1e-9:
            continue
        n = max(1, int(seg / step))
        for k in range(n):
            u = k / n
            samples.append({"t": round(t + u * seg / speed_mps, 3), "x": x0 + (x1 - x0) * u, "z": z0 + (z1 - z0) * u})
        t += seg / speed_mps
    if path:
        samples.append({"t": round(t, 3), "x": path[-1][0], "z": path[-1][1]})
    return samples


def build_mission(scene: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    scene = scene or ensure_scene()
    meta = load_meta()
    projection = CampusProjection(meta)

    # Circuito de inspección: perímetro + barrido boustrophedon sobre edificios.
    waypoints_2d = [
        (-260.0, 380.0),
        (-260.0, 250.0),
        (-260.0, 120.0),
        (-260.0, -20.0),
        (-260.0, -160.0),
        (-260.0, -300.0),
        (-160.0, -320.0),
        (-60.0, -320.0),
        (40.0, -320.0),
        (140.0, -320.0),
        (240.0, -320.0),
        (330.0, -300.0),
        (350.0, -180.0),
        (350.0, -60.0),
        (350.0, 60.0),
        (350.0, 180.0),
        (350.0, 300.0),
        (260.0, 350.0),
        (160.0, 350.0),
        (60.0, 350.0),
        (-40.0, 350.0),
        (-140.0, 350.0),
        (-220.0, 380.0),
    ]
    # Altitud variable: 38 m de crucero, 26 m en la pasada central (corredor).
    path = _catmull_rom(waypoints_2d, samples_per_segment=18)
    cruise_speed = 13.5
    samples_2d = _resample_path(path, cruise_speed, hz=10.0)
    duration = samples_2d[-1]["t"] if samples_2d else 1.0

    # Suavizado de la altitud y actitud a partir de la geometría del recorrido.
    n = len(samples_2d)
    xs = [s["x"] for s in samples_2d]
    zs = [s["z"] for s in samples_2d]
    for i, s in enumerate(samples_2d):
        u = i / max(1, n - 1)
        # Perfil de altitud: ascenso, crucero, descenso en la pasada central.
        alt = 38.0 + 8.0 * math.sin(math.pi * u) - 10.0 * math.exp(-((u - 0.52) / 0.08) ** 2)
        s["y"] = max(18.0, alt)
        # Rumbo a partir de la velocidad (mira hacia adelante).
        i0, i1 = max(0, i - 2), min(n - 1, i + 2)
        dx, dz = xs[i1] - xs[i0], zs[i1] - zs[i0]
        yaw = math.degrees(math.atan2(dx, -dz)) % 360.0
        s["yaw"] = yaw
        s["speed"] = cruise_speed
        # Alabeo proporcional al cambio de rumbo (viraje coordinado suave).
        if i > 0:
            dyaw = (yaw - samples_2d[i - 1].get("yaw", yaw) + 180.0) % 360.0 - 180.0
            s["roll"] = max(-18.0, min(18.0, dyaw * 2.4))
        else:
            s["roll"] = 0.0
        s["pitch"] = -2.0 + 3.0 * math.sin(2 * math.pi * u)

    # Corredor sin GPS: ventana temporal entre edificios.
    gps_denied = [(0.30 * duration, 0.48 * duration)]
    denied_start, denied_end = gps_denied[0]

    # Modelo de error: INS deriva por sesgo; EKF/VIO acota el error.
    samples: List[Dict[str, Any]] = []
    rng = random.Random(11)
    for i, s in enumerate(samples_2d):
        t = s["t"]
        x, z, y = s["x"], s["z"], s["y"]
        lat, lon = projection.to_geodetic(x, z)

        # Error INS: sesgo de aceleración equivalente + error de rumbo acumulado.
        ins_drift = 0.0022 * t * t
        ang = math.radians(18.0 + 0.7 * t)
        ins_x = x + ins_drift * math.cos(ang)
        ins_z = z + ins_drift * math.sin(ang)
        ins_y = y + 0.0015 * t * t

        # EKF: ruido bajo + degradación controlada en el corredor sin GPS.
        if denied_start <= t <= denied_end:
            phase = (t - denied_start) / max(1e-6, denied_end - denied_start)
            ekf_err = 0.6 + 4.5 * phase
            cov = 1.2 + 8.0 * phase
        else:
            ekf_err = 0.5
            cov = 0.9
        ekf_x = x + rng.gauss(0.0, ekf_err)
        ekf_z = z + rng.gauss(0.0, ekf_err)
        ekf_y = y + rng.gauss(0.0, ekf_err * 0.7)

        # VIO: disponible salvo huecos breves; error ~1.5 m.
        vio_ok = not (0.62 * duration < t < 0.66 * duration)
        vio_x = x + rng.gauss(0.0, 1.6)
        vio_z = z + rng.gauss(0.0, 1.6)

        err_ins = math.hypot(ins_x - x, ins_z - z)
        err_ekf = math.hypot(ekf_x - x, ekf_z - z)
        samples.append(
            {
                "t": round(t, 2),
                "lat": lat,
                "lon": lon,
                "x": round(x, 2),
                "y": round(y, 2),
                "z": round(z, 2),
                "roll": round(s["roll"], 2),
                "pitch": round(s["pitch"], 2),
                "yaw": round(s["yaw"], 2),
                "speed": round(s["speed"], 2),
                "agl": round(y, 2),
                "ins": [round(ins_x, 2), round(ins_y, 2), round(ins_z, 2)],
                "ekf": [round(ekf_x, 2), round(ekf_y, 2), round(ekf_z, 2)],
                "vo": [round(vio_x, 2), round(vio_z, 2)],
                "err_ins_m": round(err_ins, 2),
                "err_ekf_m": round(err_ekf, 2),
                "sigma_3_m": round(3.0 * cov, 2),
                "gps": not (denied_start <= t <= denied_end),
                "vio": vio_ok,
                "battery": round(max(12.0, 100.0 - 0.42 * t), 1),
                "rssi": -58 if not (denied_start <= t <= denied_end) else -92,
                "sats": 11 if not (denied_start <= t <= denied_end) else 3,
            }
        )

    mission = {
        "name": "Inspección autónoma TecMM Campus Zapopan",
        "duration": round(duration, 2),
        "hz": 10.0,
        "speed_mps": cruise_speed,
        "gps_denied": [[round(a, 2), round(b, 2)] for a, b in gps_denied],
        "waypoints": [{"x": p[0], "z": p[1]} for p in waypoints_2d],
        "pois": scene.get("pois", POIS),
        "samples": samples,
        "metrics": {
            "ins_rmse_m": round(math.sqrt(sum(s["err_ins_m"] ** 2 for s in samples) / len(samples)), 2),
            "ekf_rmse_m": round(math.sqrt(sum(s["err_ekf_m"] ** 2 for s in samples) / len(samples)), 2),
            "path_length_m": round(sum(math.hypot(xs[i] - xs[i - 1], zs[i] - zs[i - 1]) for i in range(1, n)), 1),
        },
    }
    with open(MISSION_PATH, "w", encoding="utf-8") as f:
        json.dump(mission, f, ensure_ascii=False, separators=(",", ":"))
    print(
        f"[campus] Misión guardada en {MISSION_PATH} "
        f"({len(samples)} muestras, {duration:.1f} s, INS RMSE {mission['metrics']['ins_rmse_m']:.1f} m, "
        f"EKF RMSE {mission['metrics']['ekf_rmse_m']:.1f} m)"
    )
    return mission


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera la escena y misión del campus TecMM Zapopan")
    parser.add_argument("--offline", action="store_true", help="No consultar OpenStreetMap")
    parser.add_argument("--force", action="store_true", help="Regenerar la escena aunque exista")
    parser.add_argument("--mission-only", action="store_true", help="Solo regenerar la misión")
    args = parser.parse_args()

    if args.mission_only:
        scene = ensure_scene()
    else:
        scene = ensure_scene(force=args.force, offline=args.offline)
    build_mission(scene)


if __name__ == "__main__":
    main()
