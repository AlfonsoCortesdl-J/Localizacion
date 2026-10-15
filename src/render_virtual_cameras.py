"""
render_virtual_cameras.py
=========================
Renderiza las secuencias de imágenes de las cámaras virtuales (terreno y cielo)
a partir de los metadatos de sincronización registrados durante el vuelo.

Salidas:
--------
1. data/camera_ground/img_ground_XXXXX.png:
   Fotogramas de la cámara orientada al terreno (Nadir).
2. data/camera_sky/img_sky_XXXXX.png:
   Fotogramas del rastreador estelar (Zenith / Star Tracker).
3. plots/camera_dashboard.png:
   Panel visual de ingeniería que ilustra:
   - La huella proyectada de la cámara en el ortomapa (Footprint).
   - El fotograma de terreno capturado con puntos característicos ORB.
   - La imagen del cielo estelar con detecciones astrométricas.
   - Datos de telemetría y pose 6-DOF.
"""

import os
import sys
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Importar modelos de cámaras locales
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.camera_models import VirtualGroundCamera, VirtualSkyCamera


def render_all_camera_frames(
    max_frames: int = None,  # None = renderizar TODOS los fotogramas del vuelo
    save_dashboard: bool = True
) -> None:
    """Renderiza los fotogramas de vuelo para las cámaras virtual de tierra y cielo."""
    data_dir = os.path.join(parent_dir, "data")
    plots_dir = os.path.join(parent_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    ground_csv = os.path.join(data_dir, "camera_ground_frames.csv")
    sky_csv = os.path.join(data_dir, "camera_sky_frames.csv")
    map_png = os.path.join(data_dir, "reference_map.png")
    map_meta = os.path.join(data_dir, "reference_map_meta.json")

    if not os.path.exists(ground_csv) or not os.path.exists(sky_csv):
        raise FileNotFoundError("Metadatos de cámaras no encontrados. Ejecuta primero record_flight.py.")
    if not os.path.exists(map_png) or not os.path.exists(map_meta):
        raise FileNotFoundError("Mapa de referencia no encontrado. Ejecuta primero generate_reference_map.py.")

    df_ground = pd.read_csv(ground_csv)
    df_sky = pd.read_csv(sky_csv)

    dir_ground = os.path.join(data_dir, "camera_ground")
    dir_sky = os.path.join(data_dir, "camera_sky")
    os.makedirs(dir_ground, exist_ok=True)
    os.makedirs(dir_sky, exist_ok=True)

    print("Inicializando modelos de cámaras virtuales (Nadir y Zenith)...")
    cam_ground = VirtualGroundCamera(map_png, map_meta, width=640, height=480, fov_deg=70.0)
    cam_sky = VirtualSkyCamera(width=640, height=480, fov_deg=60.0)

    # Subconjunto de fotogramas a renderizar (None = todos)
    if max_frames is None:
        indices_ground = range(len(df_ground))
    else:
        num_ground = min(len(df_ground), max_frames)
        step_ground = max(1, len(df_ground) // num_ground)
        indices_ground = range(0, len(df_ground), step_ground)[:num_ground]

    print(f"\nRenderizando {len(indices_ground)} fotogramas de Camara de Terreno en {dir_ground}...")
    sample_ground_frame = None
    sample_ground_row = None

    for idx in indices_ground:
        row = df_ground.iloc[idx]
        frame = cam_ground.render_frame(
            lat_deg=float(row["lat_geod_deg"]),
            lon_deg=float(row["lon_geod_deg"]),
            altitude_agl_m=float(row["altitude_agl_m"]),
            roll_deg=float(row["roll_deg"]),
            pitch_deg=float(row["pitch_deg"]),
            yaw_deg=float(row["yaw_deg"]),
        )
        out_filename = os.path.join(dir_ground, row["image_filename"])
        cv2.imwrite(out_filename, frame)

        # Guardar fotograma intermedio para el dashboard durante la fase de viraje
        if sample_ground_frame is None or (row["sim_time_sec"] >= 20.0 and sample_ground_row["sim_time_sec"] < 20.0):
            sample_ground_frame = frame.copy()
            sample_ground_row = row

    # Sky camera frames
    if max_frames is None:
        indices_sky = range(len(df_sky))
    else:
        num_sky = min(len(df_sky), max_frames)
        step_sky = max(1, len(df_sky) // num_sky)
        indices_sky = range(0, len(df_sky), step_sky)[:num_sky]

    print(f"Renderizando {len(indices_sky)} fotogramas de Rastreador Estelar en {dir_sky}...")
    sample_sky_frame = None
    sample_sky_row = None

    for idx in indices_sky:
        row = df_sky.iloc[idx]
        frame = cam_sky.render_frame(
            timestamp_unix=float(row["timestamp_unix"]),
            lat_deg=float(row["lat_geod_deg"]),
            lon_deg=float(row["lon_geod_deg"]),
            roll_deg=float(row["roll_deg"]),
            pitch_deg=float(row["pitch_deg"]),
            yaw_deg=float(row["yaw_deg"]),
        )
        out_filename = os.path.join(dir_sky, row["image_filename"])
        cv2.imwrite(out_filename, frame)

        if sample_sky_frame is None or (row["sim_time_sec"] >= 20.0 and sample_sky_row["sim_time_sec"] < 20.0):
            sample_sky_frame = frame.copy()
            sample_sky_row = row

    print("  [OK] Fotogramas generados y almacenados en data/camera_ground/ y data/camera_sky/")

    # 3. Generar Dashboard visual integrado
    if save_dashboard and sample_ground_frame is not None and sample_sky_frame is not None:
        generate_camera_dashboard(
            cam_ground,
            sample_ground_frame,
            sample_ground_row,
            sample_sky_frame,
            sample_sky_row,
            plots_dir
        )


def generate_camera_dashboard(
    cam_ground: VirtualGroundCamera,
    ground_frame: np.ndarray,
    ground_row: pd.Series,
    sky_frame: np.ndarray,
    sky_row: pd.Series,
    plots_dir: str
) -> str:
    """Genera un panel de visualización de ingeniería para el análisis de visión artificial."""
    print("\nGenerando panel de análisis de cámaras (plots/camera_dashboard.png)...")

    fig = plt.figure(figsize=(15, 9), facecolor="#f8f9fa")

    # 1. Panel Superior Izquierdo: Mapa Global y Posición de la Aeronave
    ax1 = fig.add_subplot(2, 2, 1)
    map_rgb = cv2.cvtColor(cam_ground.map_img, cv2.COLOR_BGR2RGB)
    ax1.imshow(map_rgb)

    # Calcular posición del avión en píxeles del mapa
    dx_m = (ground_row["lon_geod_deg"] - cam_ground.center_lon) * cam_ground.meters_per_deg_lon
    dy_m = (ground_row["lat_geod_deg"] - cam_ground.center_lat) * cam_ground.METERS_PER_DEG_LAT
    x_ac = cam_ground.map_w_px / 2.0 + (dx_m / cam_ground.gsd)
    y_ac = cam_ground.map_h_px / 2.0 - (dy_m / cam_ground.gsd)

    # Marcador de posición y flecha de rumbo
    yaw_rad = np.radians(ground_row["yaw_deg"])
    arrow_len = 120.0
    ax1.arrow(x_ac, y_ac, arrow_len * np.sin(yaw_rad), -arrow_len * np.cos(yaw_rad),
              head_width=35, head_length=45, fc="#e63946", ec="#9d0208", linewidth=2.5, zorder=5)
    ax1.plot(x_ac, y_ac, "o", color="#f1faee", markeredgecolor="#9d0208", markeredgewidth=2, markersize=10, zorder=6)

    # Huella del campo de visión (FOV Ground Footprint)
    h_m = ground_row["altitude_agl_m"]
    half_footprint_px = (h_m * np.tan(np.radians(cam_ground.fov_deg / 2.0))) / cam_ground.gsd
    footprint_box = plt.Rectangle(
        (x_ac - half_footprint_px, y_ac - half_footprint_px),
        2 * half_footprint_px, 2 * half_footprint_px,
        fill=False, edgecolor="#e63946", linestyle="--", linewidth=1.8, label="Huella FOV Nadir", zorder=4
    )
    ax1.add_patch(footprint_box)

    ax1.set_title(f"Posición en Ortomapa (t = {ground_row['sim_time_sec']:.1f} s)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Coordenada X mapa (px)", fontsize=9)
    ax1.set_ylabel("Coordenada Y mapa (px)", fontsize=9)
    ax1.legend(loc="upper right", frameon=True)

    # 2. Panel Superior Derecho: Cámara de Terreno con Detección de Características ORB
    ax2 = fig.add_subplot(2, 2, 2)
    orb = cv2.ORB_create(nfeatures=250)
    gray = cv2.cvtColor(ground_frame, cv2.COLOR_BGR2GRAY)
    kps = orb.detect(gray, None)
    img_kps = cv2.drawKeypoints(ground_frame, kps, None, color=(0, 255, 0), flags=0)
    img_kps_rgb = cv2.cvtColor(img_kps, cv2.COLOR_BGR2RGB)

    # Cruz central de la mira óptica
    h, w = ground_frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(img_kps_rgb, (cx - 25, cy), (cx + 25, cy), (255, 255, 0), 1)
    cv2.line(img_kps_rgb, (cx, cy - 25), (cx, cy + 25), (255, 255, 0), 1)

    ax2.imshow(img_kps_rgb)
    ax2.set_title(f"Cámara de Terreno (Nadir) — {len(kps)} Puntos Clave ORB Detectados", fontsize=11, fontweight="bold")
    ax2.axis("off")

    # 3. Panel Inferior Izquierdo: Rastreador Estelar (Star Tracker / Zenith)
    ax3 = fig.add_subplot(2, 2, 3)
    sky_rgb = cv2.cvtColor(sky_frame, cv2.COLOR_BGR2RGB)
    ax3.imshow(sky_rgb)
    ax3.set_title("Cámara de Cielo (Zenith / Star Tracker)", fontsize=11, fontweight="bold")
    ax3.axis("off")

    # 4. Panel Inferior Derecho: Telemetría y Pose 6-DOF
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")

    telemetry_text = (
        f"ESTADO DE LA AERONAVE (Cessna 172P)\n"
        f"----------------------------------------\n"
        f"• Tiempo de simulación : {ground_row['sim_time_sec']:.2f} s\n"
        f"• Timestamp ISO        : {ground_row['timestamp_iso']}\n"
        f"• Latitud Geodésica    : {ground_row['lat_geod_deg']:.6f}°\n"
        f"• Longitud Geodésica   : {ground_row['lon_geod_deg']:.6f}°\n"
        f"• Altitud MSL          : {ground_row['altitude_msl_m']:.1f} m\n"
        f"• Altitud AGL (Suelo)  : {ground_row['altitude_agl_m']:.1f} m\n"
        f"• Velocidad Terrestre  : {ground_row['groundspeed_mps']:.1f} m/s ({ground_row['groundspeed_mps']*1.94384:.1f} kts)\n"
        f"\n"
        f"ORIENTACIÓN Y ACTITUD 6-DOF (EULER)\n"
        f"----------------------------------------\n"
        f"• Roll / Alabeo (Φ)    : {ground_row['roll_deg']:+.2f}°\n"
        f"• Pitch / Cabeceo (θ)  : {ground_row['pitch_deg']:+.2f}°\n"
        f"• Yaw / Rumbo (ψ)      : {ground_row['yaw_deg']:05.1f}°\n"
        f"\n"
        f"PARÁMETROS ÓPTICOS\n"
        f"----------------------------------------\n"
        f"• Campo de visión (FOV): {ground_row['fov_deg']}° (Nadir) / 60° (Zenith)\n"
        f"• Resolución cámara    : 640 x 480 px\n"
        f"• GSD a esta altura    : {(ground_row['altitude_agl_m'] / cam_ground.fx):.2f} m/px\n"
    )

    ax4.text(
        0.05, 0.95, telemetry_text,
        transform=ax4.transAxes,
        fontsize=10.5,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#ffffff", edgecolor="#ced4da", linewidth=1.5)
    )

    plt.tight_layout()
    dashboard_path = os.path.join(plots_dir, "camera_dashboard.png")
    plt.savefig(dashboard_path, dpi=200)
    plt.close()
    print(f"  [OK] Dashboard de cámaras generado en: {dashboard_path}")
    return dashboard_path


if __name__ == "__main__":
    render_all_camera_frames(max_frames=None, save_dashboard=True)
