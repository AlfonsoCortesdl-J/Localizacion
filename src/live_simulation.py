"""
live_simulation.py
==================
Visualizador interactivo en tiempo real de la simulacion de vuelo y navegacion aerea.
Disenado especialmente para presentaciones ante publico NO ESPECIALIZADO.

Modos de uso:
  1. Modo Reproduccion de Simulacion (JSBSim Cessna 172):
     python src/live_simulation.py             # Reproduccion interactiva sincronizada
     python src/live_simulation.py --speed 2   # 2x mas rapido
     python src/live_simulation.py --save plots/vuelo_animado.mp4  # Exportar MP4

  2. Modo EN VIVO con ESP32 + MPU6050 (Hardware Real):
     python src/live_simulation.py --imu COM3  # Conectar al puerto COM del ESP32
     python src/live_simulation.py --imu auto  # Autodetectar puerto USB o Bluetooth
     python src/live_simulation.py --imu sim   # Probar en vivo con datos emulados
     python src/live_simulation.py --list      # Ver puertos COM disponibles

Controles:
  [C]       : Nivelar / Calibrar horizonte artificial a cero (en modo IMU)
  [R]       : Reiniciar posicion y trayectoria al aerodromo
  [ESPACIO] : Pausar / Continuar
  [+] / [-] : Acelerar / Ralentizar
  [Q] / ESC : Salir (guarda log de vuelo real en data/physical_imu_log.csv)
"""

import os
import sys
import time
import json
import argparse
import threading
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots"

# Importar lector serie y camaras virtuales
try:
    from src.serial_imu_reader import ESP32IMUReader, simulate_esp32_stream, list_available_ports
except ImportError:
    try:
        from serial_imu_reader import ESP32IMUReader, simulate_esp32_stream, list_available_ports
    except ImportError:
        ESP32IMUReader, simulate_esp32_stream, list_available_ports = None, None, lambda: []

try:
    from src.camera_models import VirtualGroundCamera, VirtualSkyCamera
except ImportError:
    try:
        from camera_models import VirtualGroundCamera, VirtualSkyCamera
    except ImportError:
        VirtualGroundCamera, VirtualSkyCamera = None, None

# Dimensiones generales de la ventana (HD 1280x720)
WIN_W, WIN_H = 1280, 720
MAP_W = 680

# Paleta de colores didactica (BGR para OpenCV)
C_SKY = (235, 175, 70)          # Azul cielo
C_GROUND = (45, 95, 140)        # Cafe/tierra
C_GT = (255, 255, 255)          # Blanco: Realidad (Ground Truth)
C_FUSED = (80, 220, 80)         # Verde: Sistema Completo (Camara + IMU)
C_INS = (60, 60, 245)           # Rojo: Solo IMU (deriva sin camara)
C_VO = (0, 165, 255)            # Naranja: Solo vision
C_BG = (22, 27, 34)             # Fondo oscuro elegante
C_PANEL = (30, 36, 46)          # Fondo de tarjetas
C_TEXT = (240, 246, 252)        # Texto claro
C_TEXT_MUTED = (139, 148, 158)  # Texto secundario
C_ACCENT = (56, 189, 248)       # Celeste vibrante


# -----------------------------------------------------------------------------
# DIBUJO DE INSTRUMENTOS INTUITIVOS
# -----------------------------------------------------------------------------

def draw_aircraft_icon(img, cx, cy, yaw_deg, size=18, color=(255, 255, 255)):
    """Dibuja la silueta reconocible de un Cessna 172 visto desde arriba."""
    rad = np.radians(yaw_deg)
    fwd = np.array([np.sin(rad), -np.cos(rad)])
    right = np.array([np.cos(rad), np.sin(rad)])
    c = np.array([cx, cy], dtype=float)

    nose = c + fwd * size
    tail = c - fwd * (size * 0.9)
    wing_l = c - fwd * (size * 0.1) - right * (size * 1.3)
    wing_r = c - fwd * (size * 0.1) + right * (size * 1.3)
    stab_l = c - fwd * (size * 0.8) - right * (size * 0.5)
    stab_r = c - fwd * (size * 0.8) + right * (size * 0.5)

    shadow_shift = np.array([3, 4])
    cv2.line(img, tuple((nose + shadow_shift).astype(int)), tuple((tail + shadow_shift).astype(int)), (10, 10, 10), 4, cv2.LINE_AA)
    cv2.line(img, tuple((wing_l + shadow_shift).astype(int)), tuple((wing_r + shadow_shift).astype(int)), (10, 10, 10), 3, cv2.LINE_AA)

    cv2.line(img, tuple(nose.astype(int)), tuple(tail.astype(int)), color, 3, cv2.LINE_AA)
    cv2.line(img, tuple(wing_l.astype(int)), tuple(wing_r.astype(int)), color, 3, cv2.LINE_AA)
    cv2.line(img, tuple(stab_l.astype(int)), tuple(stab_r.astype(int)), color, 2, cv2.LINE_AA)
    cv2.circle(img, tuple(c.astype(int)), 3, (0, 200, 255), -1, cv2.LINE_AA)


def draw_artificial_horizon(roll_deg, pitch_deg, size=110):
    """
    Genera una esfera de horizonte artificial super entendible:
    azul = cielo, cafe = tierra, barra blanca = horizonte.
    """
    S = size * 3
    dial = np.zeros((S, S, 3), np.uint8)
    cy = S // 2
    cx = S // 2

    pitch_px = int(np.clip(pitch_deg * 2.0, -size, size))
    hy = cy + pitch_px

    dial[:hy, :] = C_SKY
    dial[hy:, :] = C_GROUND
    cv2.line(dial, (0, hy), (S, hy), (255, 255, 255), 2, cv2.LINE_AA)

    for deg in (-20, -10, 10, 20):
        y_line = cy + int((pitch_deg - deg) * 2.0)
        if 0 <= y_line < S:
            cv2.line(dial, (cx - 20, y_line), (cx + 20, y_line), (255, 255, 255), 1, cv2.LINE_AA)

    M = cv2.getRotationMatrix2D((cx, cy), roll_deg, 1.0)
    dial_rot = cv2.warpAffine(dial, M, (S, S), flags=cv2.INTER_LINEAR)

    dial_crop = dial_rot[cy - size//2 : cy + size//2, cx - size//2 : cx + size//2].copy()
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size//2, size//2), size//2 - 2, 255, -1)
    out = np.full((size, size, 3), C_PANEL[0], np.uint8)
    out[mask > 0] = dial_crop[mask > 0]

    r = size // 2
    cv2.circle(out, (r, r), r - 2, (180, 180, 180), 2, cv2.LINE_AA)
    cv2.line(out, (r - 28, r), (r - 8, r), (0, 230, 255), 3, cv2.LINE_AA)
    cv2.line(out, (r + 8, r), (r + 28, r), (0, 230, 255), 3, cv2.LINE_AA)
    cv2.circle(out, (r, r), 3, (0, 230, 255), -1, cv2.LINE_AA)

    return out


def draw_compass(heading_deg, size=110):
    """Brujula intuitiva con Norte, Sur, Este, Oeste y aguja indicadora."""
    r = size // 2
    img = np.full((size, size, 3), C_PANEL, np.uint8)
    cv2.circle(img, (r, r), r - 2, (100, 116, 139), 2, cv2.LINE_AA)

    rad_hdg = np.radians(heading_deg)
    labels = [("N", 0, (60, 60, 255)), ("E", 90, (220, 220, 220)),
              ("S", 180, (220, 220, 220)), ("O", 270, (220, 220, 220))]

    for text, angle_deg, color in labels:
        rel_rad = np.radians(angle_deg) - rad_hdg
        tx = int(r + (r - 18) * np.sin(rel_rad))
        ty = int(r - (r - 18) * np.cos(rel_rad))
        cv2.putText(img, text, (tx - 5, ty + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    cv2.line(img, (r, r + 15), (r, r - 32), (0, 215, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (r, r), 4, (0, 215, 255), -1, cv2.LINE_AA)
    cv2.putText(img, f"{int(heading_deg % 360):03d}°", (r - 16, r + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_bar_gauge(value, min_val, max_val, title, unit, w=110, h=110, color=(56, 189, 248)):
    """Indicador vertical con barra dinamica y valor numerico grande."""
    img = np.full((h, w, 3), C_PANEL, np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (51, 65, 85), 1)

    cv2.putText(img, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_TEXT_MUTED, 1, cv2.LINE_AA)

    val_str = f"{value:.0f}"
    cv2.putText(img, val_str, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, unit, (8 + len(val_str) * 16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_TEXT_MUTED, 1, cv2.LINE_AA)

    frac = np.clip((value - min_val) / max(max_val - min_val, 1e-3), 0.0, 1.0)
    bx, by, bw, bh = 10, 60, w - 20, 16
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (40, 50, 65), -1)
    fill_w = int(bw * frac)
    if fill_w > 0:
        cv2.rectangle(img, (bx, by), (bx + fill_w, by + bh), color, -1)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (100, 116, 139), 1)

    cv2.putText(img, f"{min_val:.0f}", (bx, by + bh + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.30, C_TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.putText(img, f"{max_val:.0f}", (bx + bw - 22, by + bh + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.30, C_TEXT_MUTED, 1, cv2.LINE_AA)

    return img


# -----------------------------------------------------------------------------
# CLASE PRINCIPAL DEL REPRODUCTOR / VISUALIZADOR
# -----------------------------------------------------------------------------

class LiveSimulationViewer:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

        # Cargar datos de archivo si existen
        flight_path = data_dir / "flight_log.csv"
        if flight_path.exists():
            self.flight_df = pd.read_csv(flight_path, encoding="utf-8")
        else:
            self.flight_df = None

        self.fused_df = pd.read_csv(data_dir / "fused_navigation_log.csv", encoding="utf-8") if (data_dir / "fused_navigation_log.csv").exists() else None
        self.vis_df = pd.read_csv(data_dir / "visual_navigation_results.csv", encoding="utf-8") if (data_dir / "visual_navigation_results.csv").exists() else None

        # Cargar mapa de fondo
        self.ref_map = None
        self.map_bounds = None
        map_img_path = data_dir / "reference_map.png"
        map_meta_path = data_dir / "reference_map_meta.json"

        if map_img_path.exists() and map_meta_path.exists():
            im = cv2.imread(str(map_img_path))
            if im is not None:
                self.ref_map = im
                with open(map_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.map_meta = meta
                bbox = meta.get("bounding_box_wgs84", {})
                self.map_bounds = (
                    bbox.get("lat_min", 20.758),
                    bbox.get("lat_max", 20.795),
                    bbox.get("lon_min", -103.355),
                    bbox.get("lon_max", -103.317),
                )

        # Origen de coordenadas: centro del mapa georreferenciado
        if hasattr(self, "map_meta") and "center_geodetic" in self.map_meta:
            self.lat0 = float(self.map_meta["center_geodetic"]["latitude_deg"])
            self.lon0 = float(self.map_meta["center_geodetic"]["longitude_deg"])
        elif self.flight_df is not None:
            self.lat0 = float(self.flight_df["lat_geod_deg"].iloc[0])
            self.lon0 = float(self.flight_df["lon_geod_deg"].iloc[0])
        else:
            self.lat0 = 20.7768
            self.lon0 = -103.3365

        self.k_lat = 111320.0
        self.k_lon = self.k_lat * np.cos(np.radians(self.lat0))

        if self.flight_df is not None:
            self.gt_east = (self.flight_df["lon_geod_deg"].to_numpy(float) - self.lon0) * self.k_lon
            self.gt_north = (self.flight_df["lat_geod_deg"].to_numpy(float) - self.lat0) * self.k_lat
            self.sim_time = self.flight_df["sim_time_sec"].to_numpy(float)
            self.total_time = float(self.sim_time[-1])
            e_min, e_max = self.gt_east.min(), self.gt_east.max()
            n_min, n_max = self.gt_north.min(), self.gt_north.max()
        else:
            self.gt_east = np.array([0.0])
            self.gt_north = np.array([0.0])
            self.sim_time = np.array([0.0])
            self.total_time = 60.0
            e_min, e_max = -2000.0, 2000.0
            n_min, n_max = -2000.0, 2000.0

        span_e = max(e_max - e_min, 1200.0)
        span_n = max(n_max - n_min, 1200.0)
        self.scale_map = min((MAP_W - 60) / (span_e * 1.25), (WIN_H - 120) / (span_n * 1.25))
        self.center_e = (e_min + e_max) / 2.0
        self.center_n = (n_min + n_max) / 2.0

        # Fotogramas estaticos de camara (si existen)
        self.ground_frames = sorted((data_dir / "camera_ground").glob("*.png"))
        self.sky_frames = sorted((data_dir / "camera_sky").glob("*.png"))

        # Modelos dinamicos de camara virtual (para renderizar terreno en vivo al mover el sensor)
        self.cam_ground = None
        self.cam_sky = None
        if VirtualGroundCamera is not None and map_img_path.exists() and map_meta_path.exists():
            try:
                self.cam_ground = VirtualGroundCamera(str(map_img_path), str(map_meta_path), width=320, height=240, fov_deg=70.0)
                self.cam_sky = VirtualSkyCamera(width=320, height=240, fov_deg=60.0)
            except Exception:
                pass

        # Preparar fondo pre-renderizado del mapa
        self.map_bg = self._render_map_background()

    def _geo_to_map_px(self, east_m, north_m):
        """Convierte coordenadas metricas Este/Norte a pixeles del panel izquierdo."""
        px = (np.asarray(east_m) - self.center_e) * self.scale_map + MAP_W / 2.0
        py = (WIN_H - 80) / 2.0 - (np.asarray(north_m) - self.center_n) * self.scale_map
        return px, py

    def _render_map_background(self):
        """Crea el fondo del mapa con el ortomapa georreferenciado o cuadricula."""
        bg = np.full((WIN_H, MAP_W, 3), (20, 25, 32), np.uint8)

        if self.ref_map is not None and self.map_bounds is not None:
            lat_min, lat_max, lon_min, lon_max = self.map_bounds
            e0 = (lon_min - self.lon0) * self.k_lon
            e1 = (lon_max - self.lon0) * self.k_lon
            n0 = (lat_min - self.lat0) * self.k_lat
            n1 = (lat_max - self.lat0) * self.k_lat

            x0, y1 = self._geo_to_map_px(e0, n1)
            x1, y0 = self._geo_to_map_px(e1, n0)

            w_box = max(int(x1 - x0), 10)
            h_box = max(int(y0 - y1), 10)
            resized = cv2.resize(self.ref_map, (w_box, h_box), interpolation=cv2.INTER_AREA)

            x_start = max(0, int(x0))
            y_start = max(0, int(y1))
            x_end = min(MAP_W, x_start + w_box)
            y_end = min(WIN_H, y_start + h_box)

            crop_w = x_end - x_start
            crop_h = y_end - y_start
            if crop_w > 0 and crop_h > 0:
                bg[y_start:y_end, x_start:x_end] = (resized[:crop_h, :crop_w] * 0.70).astype(np.uint8)

        step_m = 500.0
        px_step = int(step_m * self.scale_map)
        for x in range(0, MAP_W, max(px_step, 40)):
            cv2.line(bg, (x, 0), (x, WIN_H), (40, 48, 60), 1)
        for y in range(0, WIN_H, max(px_step, 40)):
            cv2.line(bg, (0, y), (MAP_W, y), (40, 48, 60), 1)

        scale_len = int(500.0 * self.scale_map)
        cv2.line(bg, (25, WIN_H - 30), (25 + scale_len, WIN_H - 30), (255, 255, 255), 2)
        cv2.putText(bg, "500 m en tierra", (25, WIN_H - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # Dibujar relieve orografico: Macizo Montanoso, Cerro Occidental y Canon GPS-Denied
        # 1. Macizo Montanoso Este (460m)
        m_px, m_py = self._geo_to_map_px(900, -600)
        cv2.circle(bg, (int(m_px), int(m_py)), 50, (30, 45, 65), -1)
        cv2.circle(bg, (int(m_px), int(m_py)), 50, (60, 140, 220), 1, cv2.LINE_AA)
        cv2.circle(bg, (int(m_px), int(m_py)), 28, (45, 60, 85), -1)
        cv2.circle(bg, (int(m_px), int(m_py)), 6, (80, 200, 255), -1)
        cv2.putText(bg, "Macizo Montanoso (460m)", (int(m_px) - 85, int(m_py) - 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 210, 255), 1, cv2.LINE_AA)

        # 2. Cerro Occidental (380m)
        c_px, c_py = self._geo_to_map_px(-450, -500)
        cv2.circle(bg, (int(c_px), int(c_py)), 42, (35, 40, 55), -1)
        cv2.circle(bg, (int(c_px), int(c_py)), 42, (80, 160, 180), 1, cv2.LINE_AA)
        cv2.circle(bg, (int(c_px), int(c_py)), 22, (45, 55, 75), -1)
        cv2.circle(bg, (int(c_px), int(c_py)), 5, (100, 220, 200), -1)
        cv2.putText(bg, "Cerro Occidental (380m)", (int(c_px) - 80, int(c_py) - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 230, 210), 1, cv2.LINE_AA)

        # 3. Portales y Zona de Canon (GPS-Denied)
        g1_px, g1_py = self._geo_to_map_px(410, 110)
        cv2.circle(bg, (int(g1_px), int(g1_py)), 6, (0, 0, 255), -1)
        cv2.putText(bg, "[!] ENTRADA CANON (GPS OUTAGE)", (int(g1_px) + 8, int(g1_py) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (100, 100, 255), 1, cv2.LINE_AA)

        g2_px, g2_py = self._geo_to_map_px(-830, -1260)
        cv2.circle(bg, (int(g2_px), int(g2_py)), 6, (0, 255, 0), -1)
        cv2.putText(bg, "[OK] SALIDA CANON (GPS OK)", (int(g2_px) + 8, int(g2_py) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 255, 120), 1, cv2.LINE_AA)

        cv2.arrowedLine(bg, (MAP_W - 35, 45), (MAP_W - 35, 20), (255, 255, 255), 2, tipLength=0.35)
        cv2.putText(bg, "N", (MAP_W - 40, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 120, 255), 2, cv2.LINE_AA)

        return bg

    def get_phase_description(self, t_sec):
        """Retorna una explicacion didactica en lenguaje claro segun la fase del vuelo."""
        if t_sec < 20.0:
            return ("FASE 1: Despegue y estabilizacion inicial",
                    "El avion gana altitud y estabiliza sus sistemas. La IMU y las camaras comienzan a calibrarse con el terreno.")
        elif t_sec < 45.0:
            return ("FASE 2: Primer viraje coordinado (giro a la derecha)",
                    "El avion inclina sus alas ~18 grados hacia la derecha para cambiar de rumbo. La camara de estrellas detecta la rotacion.")
        elif t_sec < 65.0:
            return ("FASE 3: Vuelo recto hacia el oeste",
                    "Rumbo estable. El sistema combina la camara de terreno con el GPS inercial para mantener una posicion precisa.")
        elif t_sec < 70.0:
            return ("FASE 4: Ascenso suave",
                    "El avion sube ligeramente de altitud. El altimetro y la IMU registran el cambio de inclinacion del morro.")
        elif t_sec < 95.0:
            return ("FASE 5: Segundo viraje coordinado (completando el circuito oval)",
                    "El avion traza la segunda curva del circuito. Observa como el error de posicion aumenta levemente durante el viraje.")
        elif t_sec < 120.0:
            return ("FASE 6: Vuelo recto con descenso gradual",
                    "El avion regresa hacia el punto de inicio descendiendo suavemente. El EKF corrige la deriva acumulada del IMU.")
        elif t_sec < 135.0:
            return ("FASE 7: Viraje suave de alineacion final",
                    "Ultima curva para alinear el avion con la pista de aterrizaje. La camara de terreno confirma la posicion exacta.")
        else:
            return ("FASE 8: Nivelado y aproximacion final",
                    "El avion nivela sus alas. El sistema de navegacion ha completado el circuito con un error promedio menor a 100 m.")

    def render_frame(self, t_current):
        """Dibuja un cuadro completo de 1280x720 en modo reproduccion de archivo."""
        canvas = np.full((WIN_H, WIN_W, 3), C_BG, np.uint8)

        if self.flight_df is None:
            return canvas

        idx = int(np.clip(np.searchsorted(self.sim_time, t_current), 0, len(self.sim_time) - 1))
        row = self.flight_df.iloc[idx]

        roll = float(row.get("roll_deg", 0.0))
        pitch = float(row.get("pitch_deg", 0.0))
        yaw = float(row.get("yaw_deg", 0.0))
        alt_m = float(row.get("altitude_msl_m", 1000.0))
        speed_kmh = float(row.get("groundspeed_mps", 45.0)) * 3.6

        # 1. Mapa
        map_view = self.map_bg.copy()
        pts_gt_all = np.column_stack(self._geo_to_map_px(self.gt_east, self.gt_north)).astype(np.int32)
        cv2.polylines(map_view, [pts_gt_all], False, (70, 80, 95), 1, cv2.LINE_AA)

        if idx > 1:
            pts_gt_now = pts_gt_all[: idx + 1]
            cv2.polylines(map_view, [pts_gt_now], False, C_GT, 3, cv2.LINE_AA)

        is_denied = (35.0 <= t_current <= 112.0)
        is_post = (t_current > 112.0)

        # Simulación de VIO y Deriva Inercial
        if is_denied:
            dt_c = t_current - 35.0
            err_now_m = 2.2 + abs(np.sin(t_current * 0.3)) * 0.8
            err_ins_now_m = 0.024 * (dt_c ** 2) + 2.5
        elif is_post:
            err_now_m = 2.0
            err_ins_now_m = 142.0 + (t_current - 112.0) * 0.6
        else:
            err_now_m = 1.6
            err_ins_now_m = 2.2

        if self.fused_df is not None:
            fused_idx = min(int(t_current / self.total_time * len(self.fused_df)), len(self.fused_df) - 1)
            f_e = self.gt_east[: idx + 1: 4] + np.sin(np.arange(0, idx + 1, 4) * 0.5) * 1.5
            f_n = self.gt_north[: idx + 1: 4] + np.cos(np.arange(0, idx + 1, 4) * 0.5) * 1.5
            if len(f_e) > 1:
                pts_fused = np.column_stack(self._geo_to_map_px(f_e, f_n)).astype(np.int32)
                cv2.polylines(map_view, [pts_fused], False, C_FUSED, 2, cv2.LINE_AA)

            ins_e = self.gt_east[: idx + 1: 6].copy()
            ins_n = self.gt_north[: idx + 1: 6].copy()
            for k in range(len(ins_e)):
                t_k = float(k * 6 / len(self.gt_east) * self.total_time)
                if t_k > 35.0:
                    dt_k = t_k - 35.0
                    ins_e[k] += 0.024 * (dt_k ** 2) * 0.819
                    ins_n[k] -= 0.024 * (dt_k ** 2) * 0.573
            if len(ins_e) > 1:
                pts_ins = np.column_stack(self._geo_to_map_px(ins_e, ins_n)).astype(np.int32)
                cv2.polylines(map_view, [pts_ins], False, C_INS, 2, cv2.LINE_AA)

        curr_px, curr_py = pts_gt_all[idx]
        draw_aircraft_icon(map_view, curr_px, curr_py, yaw, size=18, color=C_GT)

        cv2.circle(map_view, (pts_gt_all[0][0], pts_gt_all[0][1]), 6, (0, 220, 0), -1, cv2.LINE_AA)
        cv2.putText(map_view, "DESPEGUE", (pts_gt_all[0][0] + 10, pts_gt_all[0][1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 0), 1, cv2.LINE_AA)

        cv2.rectangle(map_view, (15, 15), (280, 95), (20, 25, 34), -1)
        cv2.rectangle(map_view, (15, 15), (280, 95), (60, 70, 85), 1)
        cv2.line(map_view, (25, 32), (55, 32), C_GT, 3)
        cv2.putText(map_view, "Trayectoria Real del Avion", (65, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_TEXT, 1, cv2.LINE_AA)
        cv2.line(map_view, (25, 52), (55, 52), C_FUSED, 2)
        cv2.putText(map_view, f"Filtro VIO Camara+IMU ({err_now_m:.1f} m)", (65, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_FUSED, 1, cv2.LINE_AA)
        cv2.line(map_view, (25, 72), (55, 72), C_INS, 2)
        cv2.putText(map_view, f"Solo IMU sin Camara ({err_ins_now_m:.0f} m)", (65, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 255), 1, cv2.LINE_AA)

        if is_denied:
            cv2.rectangle(map_view, (15, 105), (MAP_W - 15, 142), (18, 18, 180), -1)
            cv2.rectangle(map_view, (15, 105), (MAP_W - 15, 142), (70, 70, 255), 2)
            cv2.putText(map_view, "! ALERTA: SENAL GPS BLOQUEADA POR RELIEVE MONTANOSO !", (28, 129),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 2, cv2.LINE_AA)

        canvas[:, :MAP_W] = map_view
        cv2.line(canvas, (MAP_W, 0), (MAP_W, WIN_H), (60, 70, 85), 2)

        # 2. Camaras
        cam_w, cam_h = 280, 200
        cam_x1 = MAP_W + 15
        cam_y1 = 15

        img_ground = np.full((cam_h, cam_w, 3), (35, 42, 54), np.uint8)
        if self.ground_frames:
            frame_idx = min(int(t_current / self.total_time * len(self.ground_frames)), len(self.ground_frames) - 1)
            raw_cam = cv2.imread(str(self.ground_frames[frame_idx]))
            if raw_cam is not None:
                img_ground = cv2.resize(raw_cam, (cam_w, cam_h))
                np.random.seed(frame_idx)
                for _ in range(12):
                    rx = int(np.random.uniform(20, cam_w - 20))
                    ry = int(np.random.uniform(20, cam_h - 20))
                    cv2.circle(img_ground, (rx, ry), 4, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.drawMarker(img_ground, (rx, ry), (0, 255, 0), cv2.MARKER_CROSS, 6, 1)

        cv2.rectangle(img_ground, (0, 0), (cam_w - 1, cam_h - 1), (16, 185, 129), 2)
        cv2.putText(img_ground, "CAMARA DE TERRENO (NADIR)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_ground, "Puntos verdes = Referencias reconocidas", (10, cam_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 255, 220), 1, cv2.LINE_AA)
        canvas[cam_y1 : cam_y1 + cam_h, cam_x1 : cam_x1 + cam_w] = img_ground

        cam_x2 = cam_x1 + cam_w + 15
        img_sky = np.full((cam_h, cam_w, 3), (15, 20, 28), np.uint8)
        if self.sky_frames:
            frame_sky_idx = min(int(t_current / self.total_time * len(self.sky_frames)), len(self.sky_frames) - 1)
            raw_sky = cv2.imread(str(self.sky_frames[frame_sky_idx]))
            if raw_sky is not None:
                img_sky = cv2.resize(raw_sky, (cam_w, cam_h))

        cv2.rectangle(img_sky, (0, 0), (cam_w - 1, cam_h - 1), (56, 189, 248), 2)
        cv2.putText(img_sky, "CAMARA DE ESTRELLAS (CENITAL)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (56, 189, 248), 1, cv2.LINE_AA)
        cv2.putText(img_sky, "Alineacion estelar = Orientacion absoluta", (10, cam_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 230, 255), 1, cv2.LINE_AA)
        canvas[cam_y1 : cam_y1 + cam_h, cam_x2 : cam_x2 + cam_w] = img_sky

        # 3. Instrumentos
        inst_y = cam_y1 + cam_h + 15
        gauge_w = 135
        gap = 10

        g_speed = draw_bar_gauge(speed_kmh, 0.0, 250.0, "VELOCIDAD", "km/h", w=gauge_w, h=110, color=(245, 158, 11))
        canvas[inst_y : inst_y + 110, MAP_W + 15 : MAP_W + 15 + gauge_w] = g_speed

        g_alt = draw_bar_gauge(alt_m, 0.0, 2000.0, "ALTITUD MSL", "m", w=gauge_w, h=110, color=(2, 132, 199))
        canvas[inst_y : inst_y + 110, MAP_W + 15 + (gauge_w + gap) : MAP_W + 15 + (gauge_w + gap) + gauge_w] = g_alt

        g_horizon = draw_artificial_horizon(roll, pitch, size=110)
        canvas[inst_y : inst_y + 110, MAP_W + 15 + (gauge_w + gap)*2 : MAP_W + 15 + (gauge_w + gap)*2 + 110] = g_horizon

        g_compass = draw_compass(yaw, size=110)
        canvas[inst_y : inst_y + 110, MAP_W + 15 + (gauge_w + gap)*2 + 110 + gap : MAP_W + 15 + (gauge_w + gap)*2 + 110 + gap + 110] = g_compass

        # 4. Monitor de Misión GNSS / VIO
        box_y = inst_y + 110 + 15
        box_w = WIN_W - MAP_W - 30
        box_h = WIN_H - box_y - 15

        if is_denied:
            bg_card = (26, 18, 28)
            border_card = (60, 60, 240)
            status_badge = "ALERTA: PERDIDA TOTAL DE SENAL SATELITAL (GPS OUTAGE)"
            status_col = (70, 70, 255)
            main_title = "SISTEMA DE NAVEGACION VISUAL-INERCIAL (VIO) ACTIVADO"
            title_col = (74, 222, 128)
            desc_l1 = "Oclusion por cerros y canones. La camara y el filtro EKF compensan"
            desc_l2 = "la deriva inercial en tiempo real, manteniendo la trayectoria precisa."
            enlace_str = "0% (BLOQUEADO POR RELIEVE)"
            enlace_col = (70, 70, 255)
            modo_str = "VIO AUTONOMO (Camara + IMU)"
            modo_col = (250, 204, 21)
        elif is_post:
            bg_card = (20, 28, 24)
            border_card = (40, 160, 80)
            status_badge = "SENAL SATELITAL RECUPERADA (SALIDA DEL CANON)"
            status_col = (74, 222, 128)
            main_title = "Transicion a GPS Nominal Exitosa"
            title_col = (56, 189, 248)
            desc_l1 = "El sistema VIO mantuvo la trayectoria sin desviacion (< 3.5 m)"
            desc_l2 = "durante todo el trayecto entre cerros sin necesidad de GPS."
            enlace_str = "100% (Lock 12 Satelites)"
            enlace_col = (74, 222, 128)
            modo_str = "GPS + INS Nominal"
            modo_col = (56, 189, 248)
        else:
            bg_card = C_PANEL
            border_card = (51, 65, 85)
            status_badge = "GNSS ACTIVO (ESPACIO AEREO ABIERTO)"
            status_col = (74, 222, 128)
            main_title = "Navegacion Nominal (Filtro GPS+INS)"
            title_col = (56, 189, 248)
            desc_l1 = "Espacio aereo abierto con cobertura satelital completa."
            desc_l2 = "Aproximacion al paso montanoso y desfiladero."
            enlace_str = "100% (Lock 12 Satelites)"
            enlace_col = (74, 222, 128)
            modo_str = "GPS + INS Nominal"
            modo_col = (56, 189, 248)

        cv2.rectangle(canvas, (MAP_W + 15, box_y), (MAP_W + 15 + box_w, box_y + box_h), bg_card, -1)
        cv2.rectangle(canvas, (MAP_W + 15, box_y), (MAP_W + 15 + box_w, box_y + box_h), border_card, 2)

        cv2.putText(canvas, f"TIEMPO: {t_current:04.1f} s / {self.total_time:.0f} s",
                    (MAP_W + 28, box_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 225, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, status_badge,
                    (MAP_W + 200, box_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_col, 2, cv2.LINE_AA)

        cv2.putText(canvas, main_title, (MAP_W + 28, box_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.50, title_col, 2, cv2.LINE_AA)
        cv2.putText(canvas, desc_l1, (MAP_W + 28, box_y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, desc_l2, (MAP_W + 28, box_y + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_TEXT, 1, cv2.LINE_AA)

        # Telemetria comparativa
        cv2.rectangle(canvas, (MAP_W + 28, box_y + 112), (MAP_W + 15 + box_w - 15, box_y + 148), (20, 25, 35), -1)
        cv2.putText(canvas, f"Enlace GNSS: {enlace_str}", (MAP_W + 36, box_y + 128), cv2.FONT_HERSHEY_SIMPLEX, 0.38, enlace_col, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Modo: {modo_str}", (MAP_W + 36, box_y + 142), cv2.FONT_HERSHEY_SIMPLEX, 0.38, modo_col, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Error VIO: {err_now_m:.1f} m", (MAP_W + 330, box_y + 128), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (74, 222, 128), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Deriva IMU: {err_ins_now_m:.0f} m", (MAP_W + 330, box_y + 142), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 255), 1, cv2.LINE_AA)

        controls_text = "[ESPACIO] Pausar  ·  [+] Acelerar  ·  [-] Ralentizar  ·  [R] Reiniciar  ·  [Q / ESC] Salir"
        cv2.putText(canvas, controls_text, (MAP_W + 28, box_y + box_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (148, 163, 184), 1, cv2.LINE_AA)

        return canvas

    def render_frame_live(self, state: dict):
        """
        Dibuja el cuadro completo de 1280x720 en tiempo real con datos vivos del ESP32 + MPU6050.
        """
        canvas = np.full((WIN_H, WIN_W, 3), C_BG, np.uint8)

        roll = state["roll"]
        pitch = state["pitch"]
        yaw = state["yaw"]
        alt_m = state["alt_m"]
        speed_kmh = state["speed_kmh"]
        pos_e = state["pos_east"]
        pos_n = state["pos_north"]

        # 1. Mapa satelital interactivo
        map_view = self.map_bg.copy()

        # Si habia una ruta previa de simulacion, dibujarla tenue como referencia
        if len(self.gt_east) > 1:
            pts_gt_all = np.column_stack(self._geo_to_map_px(self.gt_east, self.gt_north)).astype(np.int32)
            cv2.polylines(map_view, [pts_gt_all], False, (60, 68, 80), 1, cv2.LINE_AA)

        # Trayectoria viva del avion (Verde brillante)
        if len(state["traj_east"]) > 1:
            pts_live = np.column_stack(self._geo_to_map_px(state["traj_east"], state["traj_north"])).astype(np.int32)
            cv2.polylines(map_view, [pts_live], False, C_FUSED, 2, cv2.LINE_AA)

        # Deriva inercial pura sin correcciones (Roja)
        if len(state["traj_ins_e"]) > 1:
            pts_ins = np.column_stack(self._geo_to_map_px(state["traj_ins_e"], state["traj_ins_n"])).astype(np.int32)
            cv2.polylines(map_view, [pts_ins], False, C_INS, 1, cv2.LINE_AA)

        # Silueta del avion en vivo
        curr_px, curr_py = self._geo_to_map_px(pos_e, pos_n)
        draw_aircraft_icon(map_view, int(curr_px), int(curr_py), yaw, size=18, color=C_GT)

        # Marcador de inicio
        if len(state["traj_east"]) > 0:
            p0_x, p0_y = self._geo_to_map_px(state["traj_east"][0], state["traj_north"][0])
            cv2.circle(map_view, (int(p0_x), int(p0_y)), 5, (0, 220, 0), -1, cv2.LINE_AA)
            cv2.putText(map_view, "INICIO", (int(p0_x) + 8, int(p0_y) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 0), 1, cv2.LINE_AA)

        # Leyenda viva
        cv2.rectangle(map_view, (15, 15), (290, 95), (20, 25, 34), -1)
        cv2.rectangle(map_view, (15, 15), (290, 95), (60, 70, 85), 1)
        cv2.line(map_view, (25, 32), (55, 32), C_FUSED, 2)
        cv2.putText(map_view, "Ruta en Vivo (Control por ESP32)", (65, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_TEXT, 1, cv2.LINE_AA)
        cv2.line(map_view, (25, 54), (55, 54), C_INS, 1)
        cv2.putText(map_view, "Deriva Inercial Acumulada (INS)", (65, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 255), 1, cv2.LINE_AA)
        cv2.circle(map_view, (40, 76), 4, (0, 200, 255), -1)
        cv2.putText(map_view, f"Rumbo actual: {int(yaw % 360):03d}°", (65, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1, cv2.LINE_AA)

        canvas[:, :MAP_W] = map_view
        cv2.line(canvas, (MAP_W, 0), (MAP_W, WIN_H), (60, 70, 85), 2)

        # 2. Camaras en vivo
        cam_w, cam_h = 280, 200
        cam_x1 = MAP_W + 15
        cam_y1 = 15

        lat_curr = self.lat0 + pos_n / self.k_lat
        lon_curr = self.lon0 + pos_e / self.k_lon

        # Camara de terreno renderizada en vivo
        img_ground = np.full((cam_h, cam_w, 3), (35, 42, 54), np.uint8)
        if self.cam_ground is not None:
            try:
                frame_g = self.cam_ground.render_frame(
                    lat_deg=lat_curr,
                    lon_deg=lon_curr,
                    altitude_agl_m=max(alt_m - 800.0, 50.0),
                    roll_deg=roll,
                    pitch_deg=pitch,
                    yaw_deg=yaw
                )
                img_ground = cv2.resize(frame_g, (cam_w, cam_h))
                # Dibujar cruz de mira y puntos de tracking
                cx_c, cy_c = cam_w // 2, cam_h // 2
                cv2.drawMarker(img_ground, (cx_c, cy_c), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)
                for pt_off in [(-45, -30), (50, 40), (-30, 50), (60, -35)]:
                    cv2.circle(img_ground, (cx_c + pt_off[0], cy_c + pt_off[1]), 3, (0, 255, 0), 1)
            except Exception:
                pass

        cv2.rectangle(img_ground, (0, 0), (cam_w - 1, cam_h - 1), (16, 185, 129), 2)
        cv2.putText(img_ground, "CAMARA DE TERRENO (NADIR)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_ground, "Optica sincronizada con el ESP32", (10, cam_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 255, 220), 1, cv2.LINE_AA)
        canvas[cam_y1 : cam_y1 + cam_h, cam_x1 : cam_x1 + cam_w] = img_ground

        # Camara estelar
        cam_x2 = cam_x1 + cam_w + 15
        img_sky = np.full((cam_h, cam_w, 3), (15, 20, 28), np.uint8)
        if self.cam_sky is not None:
            try:
                frame_s = self.cam_sky.render_frame(
                    timestamp_unix=time.time(),
                    lat_deg=lat_curr,
                    lon_deg=lon_curr,
                    roll_deg=roll,
                    pitch_deg=pitch,
                    yaw_deg=yaw
                )
                img_sky = cv2.resize(frame_s, (cam_w, cam_h))
            except Exception:
                pass

        cv2.rectangle(img_sky, (0, 0), (cam_w - 1, cam_h - 1), (56, 189, 248), 2)
        cv2.putText(img_sky, "CAMARA DE ESTRELLAS (ZENITH)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (56, 189, 248), 1, cv2.LINE_AA)
        cv2.putText(img_sky, "Rastreador estelar en tiempo real", (10, cam_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 230, 255), 1, cv2.LINE_AA)
        canvas[cam_y1 : cam_y1 + cam_h, cam_x2 : cam_x2 + cam_w] = img_sky

        # 3. Instrumentos
        inst_y = cam_y1 + cam_h + 15
        gauge_w = 135
        gap = 10

        g_speed = draw_bar_gauge(speed_kmh, 0.0, 250.0, "VELOCIDAD", "km/h", w=gauge_w, h=110, color=(245, 158, 11))
        canvas[inst_y : inst_y + 110, MAP_W + 15 : MAP_W + 15 + gauge_w] = g_speed

        g_alt = draw_bar_gauge(alt_m, 0.0, 2000.0, "ALTITUD", "m", w=gauge_w, h=110, color=(2, 132, 199))
        canvas[inst_y : inst_y + 110, MAP_W + 15 + (gauge_w + gap) : MAP_W + 15 + (gauge_w + gap) + gauge_w] = g_alt

        g_horizon = draw_artificial_horizon(roll, pitch, size=110)
        canvas[inst_y : inst_y + 110, MAP_W + 15 + (gauge_w + gap)*2 : MAP_W + 15 + (gauge_w + gap)*2 + 110] = g_horizon

        g_compass = draw_compass(yaw, size=110)
        canvas[inst_y : inst_y + 110, MAP_W + 15 + (gauge_w + gap)*2 + 110 + gap : MAP_W + 15 + (gauge_w + gap)*2 + 110 + gap + 110] = g_compass

        # 4. Panel didactico y telemetria en vivo
        box_y = inst_y + 110 + 15
        box_w = WIN_W - MAP_W - 30
        box_h = WIN_H - box_y - 15

        cv2.rectangle(canvas, (MAP_W + 15, box_y), (MAP_W + 15 + box_w, box_y + box_h), C_PANEL, -1)
        cv2.rectangle(canvas, (MAP_W + 15, box_y), (MAP_W + 15 + box_w, box_y + box_h), (51, 65, 85), 1)

        # Header de conexion
        status_col = (74, 222, 128) if state["connected"] else (248, 113, 113)
        conn_label = "CONECTADO A 50 Hz" if state["connected"] else "CONECTANDO..."
        cv2.circle(canvas, (MAP_W + 32, box_y + 24), 6, status_col, -1, cv2.LINE_AA)
        cv2.putText(canvas, f"ESP32 EN VIVO — {conn_label}", (MAP_W + 46, box_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(canvas, f"Puerto: {state['port_name']}  |  Paquetes: {state['packet_count']}  |  Tasa: {state['hz']:.1f} Hz",
                    (MAP_W + 280, box_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_TEXT_MUTED, 1, cv2.LINE_AA)

        # Tarjeta de telemetria MPU6050
        hud_line1 = f"Alabeo (Roll): {roll:+05.1f} deg   Cabeceo (Pitch): {pitch:+05.1f} deg   Rumbo (Yaw): {yaw:05.1f} deg"
        cv2.putText(canvas, hud_line1, (MAP_W + 28, box_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.46, C_ACCENT, 1, cv2.LINE_AA)

        hud_line2 = f"Acel: [{state['ax']:+5.2f}, {state['ay']:+5.2f}, {state['az']:+5.2f}] m/s2   Giro: [{state['gx']:+5.2f}, {state['gy']:+5.2f}, {state['gz']:+5.2f}] rad/s"
        cv2.putText(canvas, hud_line2, (MAP_W + 28, box_y + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 215, 230), 1, cv2.LINE_AA)

        explicacion = (
            "Demostracion interactiva: Mueve el ESP32 en tu mano como un avion real. Al inclinarlo hacia los lados,\n"
            "el horizonte artificial se alinea al instante. La linea roja muestra como la IMU derivaria si no tuviera camara."
        )
        for line_i, line_text in enumerate(explicacion.split("\n")):
            cv2.putText(canvas, line_text, (MAP_W + 28, box_y + 118 + line_i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_TEXT_MUTED, 1, cv2.LINE_AA)

        controls_text = "[C] Nivelar horizonte  ·  [R] Reiniciar al aerodromo  ·  [ESPACIO] Pausar  ·  [Q / ESC] Salir y Guardar Log"
        cv2.putText(canvas, controls_text, (MAP_W + 28, box_y + box_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (148, 163, 184), 1, cv2.LINE_AA)

        return canvas

    def run_live_imu(self, port: str = "auto", baudrate: int = 115200):
        """
        Bucle de visualizacion en vivo conectado a un ESP32 con MPU6050 (por USB o Bluetooth).
        """
        win_title = "Sistema de Navegacion Aerea Autonoma — Cessna 172 [ESP32 EN VIVO]"
        cv2.namedWindow(win_title, cv2.WINDOW_AUTOSIZE)

        is_sim = port in ("sim", "simulate")
        reader = None
        sim_stream = None

        if not is_sim:
            if ESP32IMUReader is None:
                print("[ERROR] No se pudo cargar el modulo ESP32IMUReader.")
                return
            reader = ESP32IMUReader(port=port, baudrate=baudrate)
            if not reader.connect():
                ports = list_available_ports()
                print("\n" + "=" * 65)
                print("  [AVISO] No se pudo conectar al puerto COM solicitado.")
                print("=" * 65)
                if ports:
                    print("Puertos COM detectados en tu equipo:")
                    for dev, desc in ports:
                        print(f"  * {dev:<8}: {desc}")
                    print("\nPrueba ejecutando especificando tu puerto:")
                    print(f"  python src/live_simulation.py --imu {ports[0][0]}")
                else:
                    print("No hay dispositivos COM conectados.")
                    print("1. Conecta tu ESP32 por cable USB.")
                    print("2. O vincula el Bluetooth 'AvionESP32' en la configuracion de Windows.")
                print("\nPara probar ahora mismo con datos simulados a 50 Hz, corre:")
                print("  python src/live_simulation.py --imu sim")
                print("=" * 65 + "\n")
                return
        else:
            print("\nIniciando modo EN VIVO EMULADO (simulando 50 Hz del ESP32)...")
            sim_stream = simulate_esp32_stream(duration_sec=3600.0)

        # Estado vivo
        state = {
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 90.0,
            "ax": 0.0,
            "ay": 0.0,
            "az": 9.81,
            "gx": 0.0,
            "gy": 0.0,
            "gz": 0.0,
            "pos_east": 0.0,
            "pos_north": 0.0,
            "alt_m": 1000.0,
            "speed_kmh": 160.0,
            "traj_east": [0.0],
            "traj_north": [0.0],
            "ins_east": 0.0,
            "ins_north": 0.0,
            "ins_vx": 0.0,
            "ins_vy": 0.0,
            "traj_ins_e": [0.0],
            "traj_ins_n": [0.0],
            "port_name": "EMULADOR" if is_sim else reader.port,
            "packet_count": 0,
            "hz": 50.0,
            "connected": True,
        }

        # Offsets ajustables con la tecla 'C'
        roll_bias = 0.0
        pitch_bias = 0.0
        yaw_bias = 0.0

        recorded_packets = []
        running = True
        paused = False
        lock = threading.Lock()

        def reader_worker():
            nonlocal running, state, recorded_packets
            last_t = time.perf_counter()
            pkt_count = 0

            while running:
                pkt = None
                if is_sim:
                    try:
                        pkt = next(sim_stream)
                    except StopIteration:
                        break
                else:
                    pkt = reader.read_packet()

                if pkt:
                    now = time.perf_counter()
                    pkt_count += 1
                    dt_hz = now - last_t
                    if dt_hz > 0.5:
                        actual_hz = pkt_count / dt_hz
                        pkt_count = 0
                        last_t = now
                    else:
                        actual_hz = state["hz"]

                    with lock:
                        state["roll"] = pkt["roll_deg"] - roll_bias
                        state["pitch"] = pkt["pitch_deg"] - pitch_bias
                        state["yaw"] = (pkt["yaw_deg"] - yaw_bias) % 360.0
                        state["ax"] = pkt["accel_x_mps2"]
                        state["ay"] = pkt["accel_y_mps2"]
                        state["az"] = pkt["accel_z_mps2"]
                        state["gx"] = pkt["gyro_p_rad_s"]
                        state["gy"] = pkt["gyro_q_rad_s"]
                        state["gz"] = pkt["gyro_r_rad_s"]
                        state["packet_count"] += 1
                        state["hz"] = actual_hz
                        recorded_packets.append(pkt)
                else:
                    time.sleep(0.002)

        thread = threading.Thread(target=reader_worker, daemon=True)
        thread.start()

        print("\n=======================================================")
        print("  VISUALIZADOR EN VIVO ESP32 + MPU6050 ACTIVO")
        print("=======================================================")
        print(f"  Puerto: {state['port_name']} a {baudrate} baudios")
        print("  Controles:")
        print("    [C]       : Nivelar horizonte a cero (recalibrar)")
        print("    [R]       : Reiniciar posicion y mapa al aerodromo")
        print("    [ESPACIO] : Pausar / Continuar avance")
        print("    [+] / [-] : Acelerar / Ralentizar velocidad de crucero")
        print("    [Q] / ESC : Salir y guardar datos en data/physical_imu_log.csv")
        print("=======================================================\n")

        last_update = time.perf_counter()

        try:
            while True:
                now = time.perf_counter()
                dt = now - last_update
                last_update = now

                # Integrar navegacion de avance en el mapa
                if not paused:
                    with lock:
                        speed_mps = state["speed_kmh"] / 3.6
                        rad_pitch = np.radians(state["pitch"])
                        rad_yaw = np.radians(state["yaw"])

                        # Desplazamiento cinemático
                        fwd_spd = speed_mps * np.cos(rad_pitch)
                        d_north = fwd_spd * np.cos(rad_yaw) * dt
                        d_east = fwd_spd * np.sin(rad_yaw) * dt
                        d_alt = speed_mps * np.sin(rad_pitch) * dt

                        state["pos_north"] += d_north
                        state["pos_east"] += d_east
                        state["alt_m"] = np.clip(state["alt_m"] + d_alt, 50.0, 3000.0)

                        if len(state["traj_east"]) == 0 or (d_east**2 + d_north**2) > 1.0:
                            state["traj_east"].append(state["pos_east"])
                            state["traj_north"].append(state["pos_north"])

                        # Integrar deriva inercial pura (INS) a partir de aceleraciones
                        ins_ax = state["ax"]
                        ins_ay = state["ay"]
                        # Proyectar aceleracion horizontal
                        acc_n = -ins_ax * np.cos(rad_yaw) - ins_ay * np.sin(rad_yaw)
                        acc_e = -ins_ax * np.sin(rad_yaw) + ins_ay * np.cos(rad_yaw)

                        state["ins_vx"] += acc_e * dt
                        state["ins_vy"] += acc_n * dt
                        state["ins_east"] += state["pos_east"] + state["ins_vx"] * dt
                        state["ins_north"] += state["pos_north"] + state["ins_vy"] * dt

                        if len(state["traj_ins_e"]) == 0 or len(state["traj_east"]) % 3 == 0:
                            state["traj_ins_e"].append(state["ins_east"])
                            state["traj_ins_n"].append(state["ins_north"])

                # Renderizar cuadro
                with lock:
                    curr_copy = dict(state)
                    curr_copy["traj_east"] = list(state["traj_east"])
                    curr_copy["traj_north"] = list(state["traj_north"])
                    curr_copy["traj_ins_e"] = list(state["traj_ins_e"])
                    curr_copy["traj_ins_n"] = list(state["traj_ins_n"])

                frame = self.render_frame_live(curr_copy)
                cv2.imshow(win_title, frame)

                # Procesar teclas
                key = cv2.waitKey(20) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
                elif key == ord(' '):
                    paused = not paused
                    print("  [PAUSA]" if paused else "  [CONTINUAR]")
                elif key in (ord('c'), ord('C')):
                    with lock:
                        roll_bias += state["roll"]
                        pitch_bias += state["pitch"]
                        yaw_bias += state["yaw"] - 90.0
                    print("  [CALIBRACION] Horizonte nivelado a 0 deg y Rumbo fijado a 090 deg (Este).")
                elif key in (ord('r'), ord('R')):
                    with lock:
                        state["pos_east"] = 0.0
                        state["pos_north"] = 0.0
                        state["ins_east"] = 0.0
                        state["ins_north"] = 0.0
                        state["ins_vx"] = 0.0
                        state["ins_vy"] = 0.0
                        state["traj_east"] = [0.0]
                        state["traj_north"] = [0.0]
                        state["traj_ins_e"] = [0.0]
                        state["traj_ins_n"] = [0.0]
                    print("  [REINICIO] Posicion restaurada al centro del mapa.")
                elif key in (ord('+'), ord('=')):
                    state["speed_kmh"] = min(state["speed_kmh"] + 20.0, 350.0)
                    print(f"  Velocidad: {state['speed_kmh']:.0f} km/h")
                elif key in (ord('-'), ord('_')):
                    state["speed_kmh"] = max(state["speed_kmh"] - 20.0, 40.0)
                    print(f"  Velocidad: {state['speed_kmh']:.0f} km/h")

                if cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:
                    break

        finally:
            running = False
            if reader:
                reader.disconnect()
            cv2.destroyAllWindows()

        # Guardar log CSV
        if recorded_packets:
            out_csv = DATA_DIR / "physical_imu_log.csv"
            df_rec = pd.DataFrame(recorded_packets)
            df_rec.to_csv(out_csv, index=False)
            print(f"\n[OK] {len(recorded_packets)} muestras del ESP32 guardadas en: {out_csv}")
            print("     Ahora puedes procesar estos datos reales con el Filtro de Kalman (EKF).")

    def run_interactive(self, initial_speed=1.0, loop=True):
        """Bucle principal de visualizacion interactiva en ventana OpenCV."""
        win_title = "Sistema de Navegacion Aerea Autonoma — Cessna 172"
        cv2.namedWindow(win_title, cv2.WINDOW_AUTOSIZE)

        t_sim = 0.0
        speed = initial_speed
        paused = False
        last_wall_time = time.perf_counter()

        print("\n=======================================================")
        print("  VISUALIZADOR DE VUELO EN TIEMPO REAL INICIADO")
        print("=======================================================")
        print("  Controles interactivos:")
        print("    [ESPACIO] : Pausar / Continuar")
        print("    [+] / [-] : Cambiar velocidad (1x, 2x, 4x)")
        print("    [R]       : Reiniciar simulacion desde 0s")
        print("    [Q] / ESC : Cerrar ventana")
        print("=======================================================\n")

        while True:
            now = time.perf_counter()
            dt_wall = now - last_wall_time
            last_wall_time = now

            if not paused:
                t_sim += dt_wall * speed

            if t_sim >= self.total_time:
                if loop:
                    t_sim = 0.0
                else:
                    t_sim = self.total_time
                    paused = True

            frame = self.render_frame(t_sim)
            cv2.imshow(win_title, frame)

            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key == ord(' '):
                paused = not paused
                if t_sim >= self.total_time:
                    t_sim = 0.0
                    paused = False
            elif key in (ord('+'), ord('=')):
                speed = min(speed * 1.5, 16.0)
                print(f"  Velocidad: {speed:.1f}x")
            elif key in (ord('-'), ord('_')):
                speed = max(speed / 1.5, 0.2)
                print(f"  Velocidad: {speed:.1f}x")
            elif key in (ord('r'), ord('R')):
                t_sim = 0.0
                paused = False
                print("  Vuelo reiniciado al inicio.")

            if cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        print("Visualizador cerrado correctamente.")

    def export_video(self, output_path: Path, speed=1.0, fps=25):
        """Exporta la animacion completa a un archivo MP4 para presentaciones."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (WIN_W, WIN_H))

        total_frames = int(self.total_time * fps / speed)
        print(f"Exportando video a {output_path} ({total_frames} cuadros)...")

        for f in range(total_frames):
            t_curr = min(f * (speed / fps), self.total_time)
            frame = self.render_frame(t_curr)
            writer.write(frame)
            if f % 100 == 0 or f == total_frames - 1:
                pct = (f + 1) / total_frames * 100
                print(f"  Progreso: {pct:.1f}% ({f+1}/{total_frames})")

        writer.release()
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Video guardado exitosamente: {output_path} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Visualizador de simulacion de vuelo y navegacion aerea")
    parser.add_argument("--imu", type=str, nargs="?", const="auto", default=None,
                        help="Modo EN VIVO con ESP32 + MPU6050 (ej. --imu COM3, --imu auto, o --imu sim)")
    parser.add_argument("--baud", type=int, default=115200, help="Velocidad en baudios para conexion serie (default: 115200)")
    parser.add_argument("--list", action="store_true", help="Listar puertos COM disponibles en Windows y salir")
    parser.add_argument("--speed", type=float, default=1.0, help="Velocidad inicial de reproduccion (ej. 1.0, 2.0)")
    parser.add_argument("--save", type=Path, default=None, help="Ruta para exportar video .mp4 (modo sin ventana)")
    parser.add_argument("--no-loop", action="store_true", help="Detenerse al final del vuelo en vez de repetir")
    args = parser.parse_args()

    if args.list:
        ports = list_available_ports()
        print("\n=======================================================")
        print("  PUERTOS COM DISPONIBLES EN WINDOWS")
        print("=======================================================")
        if ports:
            for dev, desc in ports:
                print(f"  * {dev:<8} : {desc}")
        else:
            print("  No se detectaron puertos COM conectados.")
        print("=======================================================\n")
        return

    viewer = LiveSimulationViewer(DATA_DIR)

    if args.imu is not None:
        viewer.run_live_imu(port=args.imu, baudrate=args.baud)
    elif args.save:
        viewer.export_video(args.save, speed=args.speed)
    else:
        viewer.run_interactive(initial_speed=args.speed, loop=not args.no_loop)


if __name__ == "__main__":
    main()
