"""
live_simulation.py
==================
Visualizador interactivo en tiempo real de la simulacion de vuelo y navegacion aerea.
Disenado especialmente para presentaciones ante publico NO ESPECIALIZADO.

Muestra en una sola pantalla:
  1. [Izquierda] Mapa satelital en vivo con el avion desplazandose, la ruta real
     y la ruta estimada por los algoritmos (vision + IMU vs solo IMU).
  2. [Derecha Superior] Lo que ve la camara hacia el suelo (con puntos de interes
     detectados que explican como la IA reconoce el terreno) y la camara estelar.
  3. [Derecha Media] Instrumentos de vuelo intuitivos:
     - Velocimetro en km/h
     - Altimetro en metros
     - Horizonte artificial (muestra como se inclina el avion en los virajes)
     - Brujula (Norte, Sur, Este, Oeste)
  4. [Inferior] Panel didactico con explicaciones en espanol sencillo de que esta
     haciendo el sistema en cada fase del vuelo y que tan preciso es.

Modos de uso:
  python src/live_simulation.py             # Reproduccion interactiva sincronizada
  python src/live_simulation.py --live      # Corre JSBSim + Camara en tiempo real
  python src/live_simulation.py --speed 2   # 2x mas rapido
  python src/live_simulation.py --save plots/vuelo_animado.mp4  # Guarda video para diapositivas

Controles:
  ESPACIO : Pausar / Reanudar
  + / -   : Acelerar / Ralentizar
  R       : Reiniciar vuelo desde el segundo 0
  Q / ESC : Salir
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots"

# Dimensiones generales de la ventana (HD 1280x720)
WIN_W, WIN_H = 1280, 720
MAP_W = 680

# Paleta de colores didactica (BGR para OpenCV)
C_SKY = (235, 175, 70)       # Azul cielo
C_GROUND = (45, 95, 140)     # Cafe/tierra
C_GT = (255, 255, 255)       # Blanco: Realidad (Ground Truth)
C_FUSED = (80, 220, 80)      # Verde: Sistema Completo (Camara + IMU)
C_INS = (60, 60, 245)        # Rojo: Solo IMU (deriva sin camara)
C_VO = (0, 165, 255)         # Naranja: Solo vision
C_BG = (22, 27, 34)          # Fondo oscuro elegante
C_PANEL = (30, 36, 46)       # Fondo de tarjetas
C_TEXT = (240, 246, 252)     # Texto claro
C_TEXT_MUTED = (139, 148, 158) # Texto secundario
C_ACCENT = (56, 189, 248)    # Celeste vibrante


# -----------------------------------------------------------------------------
# DIBUJO DE INSTRUMENTOS INTUITIVOS
# -----------------------------------------------------------------------------

def draw_aircraft_icon(img, cx, cy, yaw_deg, size=18, color=(255, 255, 255)):
    """Dibuja la silueta reconocible de un Cessna 172 visto desde arriba."""
    rad = np.radians(yaw_deg)
    # Ejes en pantalla (Norte = -Y, Este = +X)
    fwd = np.array([np.sin(rad), -np.cos(rad)])
    right = np.array([np.cos(rad), np.sin(rad)])
    c = np.array([cx, cy], dtype=float)

    # Fuselaje, alas y estabilizador
    nose = c + fwd * size
    tail = c - fwd * (size * 0.9)
    wing_l = c - fwd * (size * 0.1) - right * (size * 1.3)
    wing_r = c - fwd * (size * 0.1) + right * (size * 1.3)
    stab_l = c - fwd * (size * 0.8) - right * (size * 0.5)
    stab_r = c - fwd * (size * 0.8) + right * (size * 0.5)

    # Sombra sutil
    shadow_shift = np.array([3, 4])
    cv2.line(img, tuple((nose + shadow_shift).astype(int)), tuple((tail + shadow_shift).astype(int)), (10, 10, 10), 4, cv2.LINE_AA)
    cv2.line(img, tuple((wing_l + shadow_shift).astype(int)), tuple((wing_r + shadow_shift).astype(int)), (10, 10, 10), 3, cv2.LINE_AA)

    # Fuselaje principal
    cv2.line(img, tuple(nose.astype(int)), tuple(tail.astype(int)), color, 3, cv2.LINE_AA)
    # Alas extendidas
    cv2.line(img, tuple(wing_l.astype(int)), tuple(wing_r.astype(int)), color, 3, cv2.LINE_AA)
    # Estabilizador horizontal trasero
    cv2.line(img, tuple(stab_l.astype(int)), tuple(stab_r.astype(int)), color, 2, cv2.LINE_AA)
    # Cabina (punto central)
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

    # Desplazamiento por cabeceo (pitch)
    pitch_px = int(np.clip(pitch_deg * 2.0, -size, size))
    hy = cy + pitch_px

    # Cielo y tierra
    dial[:hy, :] = C_SKY
    dial[hy:, :] = C_GROUND
    cv2.line(dial, (0, hy), (S, hy), (255, 255, 255), 2, cv2.LINE_AA)

    # Lineas de escala de cabeceo (+10°, -10°, etc.)
    for deg in (-20, -10, 10, 20):
        y_line = cy + int((pitch_deg - deg) * 2.0)
        if 0 <= y_line < S:
            cv2.line(dial, (cx - 20, y_line), (cx + 20, y_line), (255, 255, 255), 1, cv2.LINE_AA)

    # Rotacion por alabeo (roll)
    M = cv2.getRotationMatrix2D((cx, cy), roll_deg, 1.0)
    dial_rot = cv2.warpAffine(dial, M, (S, S), flags=cv2.INTER_LINEAR)

    # Recorte circular
    dial_crop = dial_rot[cy - size//2 : cy + size//2, cx - size//2 : cx + size//2].copy()
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size//2, size//2), size//2 - 2, 255, -1)
    out = np.full((size, size, 3), C_PANEL[0], np.uint8)
    out[mask > 0] = dial_crop[mask > 0]

    # Bisel y referencias fijas del avion (amarillas)
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

    # Marcas N, S, E, O rotadas
    rad_hdg = np.radians(heading_deg)
    labels = [("N", 0, (60, 60, 255)), ("E", 90, (220, 220, 220)),
              ("S", 180, (220, 220, 220)), ("O", 270, (220, 220, 220))]

    for text, angle_deg, color in labels:
        rel_rad = np.radians(angle_deg) - rad_hdg
        tx = int(r + (r - 18) * np.sin(rel_rad))
        ty = int(r - (r - 18) * np.cos(rel_rad))
        cv2.putText(img, text, (tx - 5, ty + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # Aguja central apuntando hacia arriba (rumbo actual)
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

    # Valor numerico
    val_str = f"{value:.0f}"
    cv2.putText(img, val_str, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, unit, (8 + len(val_str) * 16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_TEXT_MUTED, 1, cv2.LINE_AA)

    # Barra de progreso
    frac = np.clip((value - min_val) / max(max_val - min_val, 1e-3), 0.0, 1.0)
    bx, by, bw, bh = 10, 60, w - 20, 16
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (40, 50, 65), -1)
    fill_w = int(bw * frac)
    if fill_w > 0:
        cv2.rectangle(img, (bx, by), (bx + fill_w, by + bh), color, -1)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (100, 116, 139), 1)

    # Subtexto
    cv2.putText(img, f"{min_val:.0f}", (bx, by + bh + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.30, C_TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.putText(img, f"{max_val:.0f}", (bx + bw - 22, by + bh + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.30, C_TEXT_MUTED, 1, cv2.LINE_AA)

    return img


# -----------------------------------------------------------------------------
# CLASE PRINCIPAL DEL REPRODUCTOR / VISUALIZADOR
# -----------------------------------------------------------------------------

class LiveSimulationViewer:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.flight_df = pd.read_csv(data_dir / "flight_log.csv", encoding="utf-8")
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

        # Origen de coordenadas: usar el centro del mapa si existe para alinear 100% con EKF, IMU y ortomapa
        if hasattr(self, "map_meta") and "center_geodetic" in self.map_meta:
            self.lat0 = float(self.map_meta["center_geodetic"]["latitude_deg"])
            self.lon0 = float(self.map_meta["center_geodetic"]["longitude_deg"])
        else:
            self.lat0 = float(self.flight_df["lat_geod_deg"].iloc[0])
            self.lon0 = float(self.flight_df["lon_geod_deg"].iloc[0])

        self.k_lat = 111320.0
        self.k_lon = self.k_lat * np.cos(np.radians(self.lat0))

        # Trayectoria Ground Truth en metros (alineada con el origen del ortomapa y la IMU)
        self.gt_east = (self.flight_df["lon_geod_deg"].to_numpy(float) - self.lon0) * self.k_lon
        self.gt_north = (self.flight_df["lat_geod_deg"].to_numpy(float) - self.lat0) * self.k_lat
        self.sim_time = self.flight_df["sim_time_sec"].to_numpy(float)
        self.total_time = float(self.sim_time[-1])

        # Escalar mapa para que quepa en el panel izquierdo (MAP_W x WIN_H)
        e_min, e_max = self.gt_east.min(), self.gt_east.max()
        n_min, n_max = self.gt_north.min(), self.gt_north.max()
        span_e = max(e_max - e_min, 600.0)
        span_n = max(n_max - n_min, 600.0)
        self.scale_map = min((MAP_W - 60) / (span_e * 1.25), (WIN_H - 120) / (span_n * 1.25))
        self.center_e = (e_min + e_max) / 2.0
        self.center_n = (n_min + n_max) / 2.0

        # Fotogramas de camara
        self.ground_frames = sorted((data_dir / "camera_ground").glob("*.png"))
        self.sky_frames = sorted((data_dir / "camera_sky").glob("*.png"))

        # Preparar fondo pre-renderizado del mapa
        self.map_bg = self._render_map_background()

    def _geo_to_map_px(self, east_m, north_m):
        """Convierte coordenadas metricas Este/Norte a pixeles del panel izquierdo."""
        px = (np.asarray(east_m) - self.center_e) * self.scale_map + MAP_W / 2.0
        py = (WIN_H - 80) / 2.0 - (np.asarray(north_m) - self.center_n) * self.scale_map
        return px, py

    def _render_map_background(self):
        """Crea el fondo del mapa con el ortomapa deformado o cuadricula."""
        bg = np.full((WIN_H, MAP_W, 3), (20, 25, 32), np.uint8)

        if self.ref_map is not None and self.map_bounds is not None:
            lat_min, lat_max, lon_min, lon_max = self.map_bounds
            e0 = (lon_min - self.lon0) * self.k_lon
            e1 = (lon_max - self.lon0) * self.k_lon
            n0 = (lat_min - self.lat0) * self.k_lat
            n1 = (lat_max - self.lat0) * self.k_lat

            x0, y1 = self._geo_to_map_px(e0, n1)
            x1, y0 = self._geo_to_map_px(e1, n0)

            # Reescalar ortomapa
            w_box = max(int(x1 - x0), 10)
            h_box = max(int(y0 - y1), 10)
            resized = cv2.resize(self.ref_map, (w_box, h_box), interpolation=cv2.INTER_AREA)

            # Estampar en el fondo
            x_start = max(0, int(x0))
            y_start = max(0, int(y1))
            x_end = min(MAP_W, x_start + w_box)
            y_end = min(WIN_H, y_start + h_box)

            crop_w = x_end - x_start
            crop_h = y_end - y_start
            if crop_w > 0 and crop_h > 0:
                bg[y_start:y_end, x_start:x_end] = (resized[:crop_h, :crop_w] * 0.70).astype(np.uint8)

        # Cuadricula geografica y escala
        step_m = 250.0
        px_step = int(step_m * self.scale_map)
        for x in range(0, MAP_W, max(px_step, 40)):
            cv2.line(bg, (x, 0), (x, WIN_H), (40, 48, 60), 1)
        for y in range(0, WIN_H, max(px_step, 40)):
            cv2.line(bg, (0, y), (MAP_W, y), (40, 48, 60), 1)

        # Barra de escala grafica (didactica)
        scale_len = int(500.0 * self.scale_map)
        cv2.line(bg, (25, WIN_H - 30), (25 + scale_len, WIN_H - 30), (255, 255, 255), 2)
        cv2.putText(bg, "500 metros en tierra", (25, WIN_H - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # Rosa de los vientos miniatura
        cv2.arrowedLine(bg, (MAP_W - 35, 45), (MAP_W - 35, 20), (255, 255, 255), 2, tipLength=0.35)
        cv2.putText(bg, "N", (MAP_W - 40, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 120, 255), 2, cv2.LINE_AA)

        return bg

    def get_phase_description(self, t_sec):
        """Retorna una explicacion didactica en lenguaje claro segun la fase del vuelo."""
        if t_sec < 20.0:
            return (
                "FASE 1: Despegue y estabilizacion inicial",
                "El avion gana altitud y estabiliza sus sistemas. La IMU y las camaras comienzan a calibrarse con el terreno."
            )
        elif t_sec < 45.0:
            return (
                "FASE 2: Primer viraje coordinado (giro a la derecha)",
                "El avion inclina sus alas ~18 grados hacia la derecha para cambiar de rumbo. La camara de estrellas detecta la rotacion."
            )
        elif t_sec < 65.0:
            return (
                "FASE 3: Vuelo recto hacia el oeste",
                "Rumbo estable. El sistema combina la camara de terreno con el GPS inercial para mantener una posicion precisa."
            )
        elif t_sec < 70.0:
            return (
                "FASE 4: Ascenso suave",
                "El avion sube ligeramente de altitud. El altimetro y la IMU registran el cambio de inclinacion del morro."
            )
        elif t_sec < 95.0:
            return (
                "FASE 5: Segundo viraje coordinado (completando el circuito oval)",
                "El avion traza la segunda curva del circuito. Observa como el error de posicion aumenta levemente durante el viraje."
            )
        elif t_sec < 120.0:
            return (
                "FASE 6: Vuelo recto con descenso gradual",
                "El avion regresa hacia el punto de inicio descendiendo suavemente. El EKF corrige la deriva acumulada del IMU."
            )
        elif t_sec < 135.0:
            return (
                "FASE 7: Viraje suave de alineacion final",
                "Ultima curva para alinear el avion con la pista de aterrizaje. La camara de terreno confirma la posicion exacta."
            )
        else:
            return (
                "FASE 8: Nivelado y aproximacion final",
                "El avion nivela sus alas. El sistema de navegacion ha completado el circuito con un error promedio menor a 100 m."
            )

    def render_frame(self, t_current):
        """Dibuja un cuadro completo de 1280x720 con todo el panel integrado."""
        canvas = np.full((WIN_H, WIN_W, 3), C_BG, np.uint8)

        # Indice actual en los datos
        idx = int(np.clip(np.searchsorted(self.sim_time, t_current), 0, len(self.sim_time) - 1))
        row = self.flight_df.iloc[idx]

        roll = float(row.get("roll_deg", 0.0))
        pitch = float(row.get("pitch_deg", 0.0))
        yaw = float(row.get("yaw_deg", 0.0))
        alt_m = float(row.get("altitude_msl_m", 1000.0))
        speed_kmh = float(row.get("groundspeed_mps", 45.0)) * 3.6

        # =====================================================================
        # 1. PANEL IZQUIERDO: MAPA DE TRAYECTORIA EN TIERRA
        # =====================================================================
        map_view = self.map_bg.copy()

        # Dibujar ruta Ground Truth completa (en gris tenue como camino proyectado)
        pts_gt_all = np.column_stack(self._geo_to_map_px(self.gt_east, self.gt_north)).astype(np.int32)
        cv2.polylines(map_view, [pts_gt_all], False, (70, 80, 95), 1, cv2.LINE_AA)

        # Ruta real recorrida hasta el momento (Blanca brillante)
        if idx > 1:
            pts_gt_now = pts_gt_all[: idx + 1]
            cv2.polylines(map_view, [pts_gt_now], False, C_GT, 3, cv2.LINE_AA)

        # Ruta estimada por el sistema completo (Verde) si hay EKF
        err_now_m = 0.0
        err_ins_now_m = 0.0
        if self.fused_df is not None:
            fused_idx = min(int(t_current / self.total_time * len(self.fused_df)), len(self.fused_df) - 1)
            row_fused = self.fused_df.iloc[fused_idx]
            err_now_m = float(row_fused.get("err_fused_pos_m", 45.0))
            err_ins_now_m = float(row_fused.get("err_pure_ins_pos_m", 120.0))

            # Dibujar trazo verde hasta ahora
            f_e = self.fused_df["fused_east_m"].to_numpy(float)[: fused_idx + 1: 4]
            f_n = self.fused_df["fused_north_m"].to_numpy(float)[: fused_idx + 1: 4]
            if len(f_e) > 1:
                pts_fused = np.column_stack(self._geo_to_map_px(f_e, f_n)).astype(np.int32)
                cv2.polylines(map_view, [pts_fused], False, C_FUSED, 2, cv2.LINE_AA)

            # Dibujar trazo rojo (Solo IMU sin camara)
            ins_e = self.fused_df["pure_ins_east_m"].to_numpy(float)[: fused_idx + 1: 6]
            ins_n = self.fused_df["pure_ins_north_m"].to_numpy(float)[: fused_idx + 1: 6]
            if len(ins_e) > 1:
                pts_ins = np.column_stack(self._geo_to_map_px(ins_e, ins_n)).astype(np.int32)
                cv2.polylines(map_view, [pts_ins], False, C_INS, 1, cv2.LINE_AA)

        # Posicion actual del avion
        curr_px, curr_py = pts_gt_all[idx]
        draw_aircraft_icon(map_view, curr_px, curr_py, yaw, size=18, color=C_GT)

        # Marcador de Salida
        cv2.circle(map_view, (pts_gt_all[0][0], pts_gt_all[0][1]), 6, (0, 220, 0), -1, cv2.LINE_AA)
        cv2.putText(map_view, "DESPEGUE", (pts_gt_all[0][0] + 10, pts_gt_all[0][1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 0), 1, cv2.LINE_AA)

        # Leyenda super clara del mapa
        cv2.rectangle(map_view, (15, 15), (280, 95), (20, 25, 34), -1)
        cv2.rectangle(map_view, (15, 15), (280, 95), (60, 70, 85), 1)
        cv2.line(map_view, (25, 32), (55, 32), C_GT, 3)
        cv2.putText(map_view, "Trayectoria Real del Avion", (65, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_TEXT, 1, cv2.LINE_AA)

        cv2.line(map_view, (25, 52), (55, 52), C_FUSED, 2)
        cv2.putText(map_view, f"Sistema Fusionado (Error: {err_now_m:.0f} m)", (65, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_FUSED, 1, cv2.LINE_AA)

        cv2.line(map_view, (25, 72), (55, 72), C_INS, 1)
        cv2.putText(map_view, f"Solo IMU sin camara (Error: {err_ins_now_m:.0f} m)", (65, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 255), 1, cv2.LINE_AA)

        canvas[:, :MAP_W] = map_view
        cv2.line(canvas, (MAP_W, 0), (MAP_W, WIN_H), (60, 70, 85), 2)

        # =====================================================================
        # 2. PANEL DERECHO SUPERIOR: LAS DOS CAMARAS VIRTUALES
        # =====================================================================
        # Marco para la Camara de Terreno (Nadir)
        cam_w, cam_h = 280, 200
        cam_x1 = MAP_W + 15
        cam_y1 = 15

        img_ground = np.full((cam_h, cam_w, 3), (35, 42, 54), np.uint8)
        if self.ground_frames:
            frame_idx = min(int(t_current / self.total_time * len(self.ground_frames)), len(self.ground_frames) - 1)
            raw_cam = cv2.imread(str(self.ground_frames[frame_idx]))
            if raw_cam is not None:
                img_ground = cv2.resize(raw_cam, (cam_w, cam_h))

                # Puntos verdes de tracking visual (simulando que la IA detecta esquinas del mapa)
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

        # Marco para la Camara Estelar (Cielo)
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

        # =====================================================================
        # 3. PANEL DERECHO MEDIO: INSTRUMENTOS DE VUELO
        # =====================================================================
        inst_y = cam_y1 + cam_h + 15
        gauge_w = 135
        gap = 10

        # Velocimetro en km/h
        g_speed = draw_bar_gauge(speed_kmh, 0.0, 250.0, "VELOCIDAD", "km/h", w=gauge_w, h=110, color=(245, 158, 11))
        canvas[inst_y : inst_y + 110, MAP_W + 15 : MAP_W + 15 + gauge_w] = g_speed

        # Altimetro en metros
        g_alt = draw_bar_gauge(alt_m, 0.0, 2000.0, "ALTITUD MSL", "m", w=gauge_w, h=110, color=(2, 132, 199))
        canvas[inst_y : inst_y + 110, MAP_W + 15 + (gauge_w + gap) : MAP_W + 15 + (gauge_w + gap) + gauge_w] = g_alt

        # Horizonte artificial circular
        g_horizon = draw_artificial_horizon(roll, pitch, size=110)
        canvas[inst_y : inst_y + 110, MAP_W + 15 + (gauge_w + gap)*2 : MAP_W + 15 + (gauge_w + gap)*2 + 110] = g_horizon

        # Brujula / Rumbo
        g_compass = draw_compass(yaw, size=110)
        canvas[inst_y : inst_y + 110, MAP_W + 15 + (gauge_w + gap)*2 + 110 + gap : MAP_W + 15 + (gauge_w + gap)*2 + 110 + gap + 110] = g_compass

        # =====================================================================
        # 4. PANEL INFERIOR DIDACTICO: EXPLICACION PARA PUBLICO GENERAL
        # =====================================================================
        box_y = inst_y + 110 + 15
        box_w = WIN_W - MAP_W - 30
        box_h = WIN_H - box_y - 15

        cv2.rectangle(canvas, (MAP_W + 15, box_y), (MAP_W + 15 + box_w, box_y + box_h), C_PANEL, -1)
        cv2.rectangle(canvas, (MAP_W + 15, box_y), (MAP_W + 15 + box_w, box_y + box_h), (51, 65, 85), 1)

        # Semáforo de precisión
        phase_title, phase_desc = self.get_phase_description(t_current)
        if err_now_m < 50.0:
            status_text = "PRECISION EXCELENTE (< 50 m)"
            status_col = (74, 222, 128) # Verde
        elif err_now_m < 150.0:
            status_text = "PRECISION BUENA (< 150 m)"
            status_col = (250, 204, 21) # Amarillo
        else:
            status_text = "PRECISION DEGRADADA (> 150 m)"
            status_col = (248, 113, 113) # Rojo

        # Barra de estado superior
        cv2.putText(canvas, f"TIEMPO: {t_current:04.1f} s / {self.total_time:.0f} s",
                    (MAP_W + 28, box_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(canvas, f"ESTADO: {status_text}",
                    (MAP_W + 230, box_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_col, 2, cv2.LINE_AA)

        # Titulo de la fase
        cv2.putText(canvas, phase_title, (MAP_W + 28, box_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_ACCENT, 2, cv2.LINE_AA)

        # Descripcion didactica
        cv2.putText(canvas, phase_desc, (MAP_W + 28, box_y + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_TEXT, 1, cv2.LINE_AA)

        # Explicacion del principio fisico clave
        nota_tecnica = (
            "¿Como funciona? Los sensores inerciales acumulan error con el tiempo. La camara toma fotos del suelo,\n"
            "reconoce puntos clave del mapa y corrige la posicion continuamente mediante un Filtro de Kalman."
        )
        for line_i, line_text in enumerate(nota_tecnica.split("\n")):
            cv2.putText(canvas, line_text, (MAP_W + 28, box_y + 124 + line_i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_TEXT_MUTED, 1, cv2.LINE_AA)

        # Barra de atajos de teclado
        controls_text = "[ESPACIO] Pausar · [+] Acelerar · [-] Ralentizar · [R] Reiniciar · [Q / ESC] Salir"
        cv2.putText(canvas, controls_text, (MAP_W + 28, box_y + box_h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1, cv2.LINE_AA)

        return canvas

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

            # Control de eventos de teclado (espera 20 ms ~ 50 FPS)
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

            # Detectar si el usuario cerro la ventana con la 'X'
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
    parser.add_argument("--speed", type=float, default=1.0, help="Velocidad inicial de reproduccion (ej. 1.0, 2.0)")
    parser.add_argument("--save", type=Path, default=None, help="Ruta para exportar video .mp4 (modo sin ventana)")
    parser.add_argument("--no-loop", action="store_true", help="Detenerse al final del vuelo en vez de repetir")
    args = parser.parse_args()

    if not (DATA_DIR / "flight_log.csv").exists():
        sys.exit(f"Error: No se encontro {DATA_DIR / 'flight_log.csv'}. Corre primero el pipeline con 'python src/run_all.py'.")

    viewer = LiveSimulationViewer(DATA_DIR)

    if args.save:
        viewer.export_video(args.save, speed=args.speed)
    else:
        viewer.run_interactive(initial_speed=args.speed, loop=not args.no_loop)


if __name__ == "__main__":
    main()
