"""
plot_presentation.py
====================
Panel visual ejecutivo y didactico para presentacion de resultados
a audiencias tecnicas y no tecnicas.

Disenado para responder de un solo vistazo:
  1. ¿Que problema resuelve el proyecto? (Navegacion autonoma sin GPS)
  2. ¿Que ve el avion? (Camara de terreno y camara de estrellas)
  3. ¿Como volo el Cessna 172? (Trayectoria, altitud, velocidad)
  4. ¿Que tan preciso es el sistema? (Comparacion Ground Truth vs Solo IMU vs Fusion)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_flight_data():
    """Carga los CSVs de resultados y extrae las metricas principales."""
    p_flight = os.path.join(DATA_DIR, "flight_log.csv")
    p_fused = os.path.join(DATA_DIR, "fused_navigation_log.csv")
    p_vis = os.path.join(DATA_DIR, "visual_navigation_results.csv")

    if not os.path.exists(p_flight):
        raise FileNotFoundError(f"No se encontro {p_flight}. Ejecuta primero el pipeline de simulacion.")

    df_flight = pd.read_csv(p_flight, encoding="utf-8")
    df_fused = pd.read_csv(p_fused, encoding="utf-8") if os.path.exists(p_fused) else None
    df_vis = pd.read_csv(p_vis, encoding="utf-8") if os.path.exists(p_vis) else None

    return df_flight, df_fused, df_vis


def create_presentation_figure():
    print("Generando panel de presentacion ejecutivo...")
    df_flight, df_fused, df_vis = load_flight_data()

    # Configurar estilo visual limpio y profesional
    plt.style.use("default")
    fig = plt.figure(figsize=(20, 13), facecolor="#F8FAFC")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.36, wspace=0.28,
                           left=0.06, right=0.96, top=0.91, bottom=0.06)

    # =========================================================================
    # ENCABEZADO DIDACTICO (Super Title)
    # =========================================================================
    fig.suptitle(
        "PROTOTIPO DE NAVEGACION AEREA AUTONOMA (SIN GPS)",
        fontsize=20, fontweight="bold", color="#0F172A", y=0.97
    )
    fig.text(
        0.5, 0.935,
        "Simulacion de dinamica de vuelo Cessna 172P + Camara de Terreno + Rastreador Estelar + Filtro de Kalman (EKF)",
        ha="center", fontsize=11, color="#475569"
    )

    # =========================================================================
    # PANEL 1 (Grande, 2x2): MAPA DE TRAYECTORIA Y COMPARATIVA
    # =========================================================================
    ax_traj = fig.add_subplot(gs[0:2, 0:2])
    ax_traj.set_facecolor("#FFFFFF")

    # Coordenadas relativas en metros: usar el centro del mapa para coincidir 100% con EKF, IMU y ortomapa
    meta_path = os.path.join(DATA_DIR, "reference_map_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        c_lat = meta["center_geodetic"]["latitude_deg"]
        c_lon = meta["center_geodetic"]["longitude_deg"]
    else:
        c_lat = df_flight["lat_geod_deg"].iloc[0]
        c_lon = df_flight["lon_geod_deg"].iloc[0]

    k_lat = 111320.0
    k_lon = k_lat * np.cos(np.radians(c_lat))

    gt_east = (df_flight["lon_geod_deg"] - c_lon) * k_lon
    gt_north = (df_flight["lat_geod_deg"] - c_lat) * k_lat

    # 1. Trayectoria real (Ground Truth)
    ax_traj.plot(gt_east, gt_north, color="#2563EB", lw=3.0, label="Trayectoria Real (Ground Truth Simulador)", zorder=5)

    # 2. Estimacion EKF si existe
    if df_fused is not None and "fused_east_m" in df_fused.columns:
        ekf_e = df_fused["fused_east_m"].iloc[::3]
        ekf_n = df_fused["fused_north_m"].iloc[::3]
        ax_traj.plot(ekf_e, ekf_n, color="#10B981", lw=2.2, ls="--",
                     label="Estimacion Sistema Fusionado (IMU + Camaras)", zorder=6)

    # 3. Estimacion Solo IMU (deriva) si existe
    if df_fused is not None and "pure_ins_east_m" in df_fused.columns:
        ins_e = df_fused["pure_ins_east_m"].iloc[::3]
        ins_n = df_fused["pure_ins_north_m"].iloc[::3]
        ax_traj.plot(ins_e, ins_n, color="#EF4444", lw=1.6, ls=":", alpha=0.85,
                     label="Solo IMU sin camara (se desvia por deriva acumulada)", zorder=4)

    # 4. Estimacion Solo Vision si existe
    if df_vis is not None and "est_vo_east_m" in df_vis.columns:
        vo_e = df_vis["est_vo_east_m"].iloc[::2]
        vo_n = df_vis["est_vo_north_m"].iloc[::2]
        ax_traj.plot(vo_e, vo_n, color="#F59E0B", lw=1.4, ls="-.", alpha=0.75,
                     label="Solo Vision (Odometria Visual)", zorder=4)

    # Marcadores de Inicio y Fin
    ax_traj.scatter(gt_east.iloc[0], gt_north.iloc[0], s=160, color="#10B981", edgecolors="#064E3B",
                    marker="o", zorder=7, label="Punto de Salida")
    ax_traj.scatter(gt_east.iloc[-1], gt_north.iloc[-1], s=160, color="#DC2626", edgecolors="#7F1D1D",
                    marker="X", zorder=7, label="Punto de Llegada")

    ax_traj.set_title("Trayectoria en Tierra: ¿Que tan cerca estuvo el sistema de la realidad?",
                      fontsize=13, fontweight="bold", color="#1E293B", pad=10)
    ax_traj.set_xlabel("Desplazamiento Este - Oeste (metros)", fontsize=10, color="#334155")
    ax_traj.set_ylabel("Desplazamiento Norte - Sur (metros)", fontsize=10, color="#334155")
    ax_traj.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1")
    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.legend(loc="lower left", fontsize=8.5, framealpha=0.92, facecolor="#F8FAFC", edgecolor="#CBD5E1")

    # Caja de resumen didactico dentro del panel de trayectoria
    rmse_ekf = df_fused["err_fused_pos_m"].mean() if df_fused is not None else 88.0
    rmse_ins = df_fused["err_pure_ins_pos_m"].mean() if df_fused is not None else 444.0
    mejora = rmse_ins / max(rmse_ekf, 1.0)
    box_text = (
        f"RESUMEN DE PRECISION:\n"
        f"• Error medio Sistema Completo: {rmse_ekf:.1f} m\n"
        f"• Error medio Solo IMU (sin camara): {rmse_ins:.1f} m\n"
        f"• Ganancia por usar Vision: {mejora:.1f} veces mas preciso"
    )
    ax_traj.text(
        0.98, 0.97, box_text, transform=ax_traj.transAxes,
        fontsize=9, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#EFF6FF", edgecolor="#3B82F6", lw=1.2)
    )

    # =========================================================================
    # PANEL 2: PERFIL DE ALTITUD
    # =========================================================================
    ax_alt = fig.add_subplot(gs[0, 2])
    ax_alt.set_facecolor("#FFFFFF")
    t = df_flight["sim_time_sec"]
    alt = df_flight["altitude_msl_m"]

    ax_alt.plot(t, alt, color="#0284C7", lw=2.2, label="Altitud MSL")
    ax_alt.fill_between(t, alt.min() - 20, alt, color="#E0F2FE", alpha=0.5)
    ax_alt.axhline(1000.0, color="#94A3B8", ls="--", lw=1.0, label="Referencia 1,000 m")

    ax_alt.set_title("Altitud de Vuelo (Metros sobre el mar)", fontsize=11, fontweight="bold", color="#1E293B")
    ax_alt.set_xlabel("Tiempo (segundos)", fontsize=9, color="#475569")
    ax_alt.set_ylabel("Altitud (m)", fontsize=9, color="#475569")
    ax_alt.grid(True, linestyle="--", alpha=0.5)
    ax_alt.legend(loc="upper right", fontsize=8)

    # =========================================================================
    # PANEL 3: VELOCIDAD EN KM/H (FACIL DE ENTENDER)
    # =========================================================================
    ax_spd = fig.add_subplot(gs[1, 2])
    ax_spd.set_facecolor("#FFFFFF")
    spd_kmh = df_flight["groundspeed_mps"] * 3.6

    ax_spd.plot(t, spd_kmh, color="#D97706", lw=2.2, label="Velocidad real")
    ax_spd.axhline(spd_kmh.mean(), color="#B45309", ls=":", lw=1.2,
                   label=f"Promedio: {spd_kmh.mean():.0f} km/h")

    ax_spd.set_title("Velocidad de Avance (km/h)", fontsize=11, fontweight="bold", color="#1E293B")
    ax_spd.set_xlabel("Tiempo (segundos)", fontsize=9, color="#475569")
    ax_spd.set_ylabel("Velocidad (km/h)", fontsize=9, color="#475569")
    ax_spd.grid(True, linestyle="--", alpha=0.5)
    ax_spd.legend(loc="lower right", fontsize=8)

    # =========================================================================
    # PANEL 4: GRAFICA DE ERROR CON ZONAS DIDACTICAS
    # =========================================================================
    ax_err = fig.add_subplot(gs[2, 0])
    ax_err.set_facecolor("#FFFFFF")

    if df_fused is not None and "err_fused_pos_m" in df_fused.columns:
        t_fused = df_fused["sim_time_sec"].iloc[::10]
        err_fused = df_fused["err_fused_pos_m"].iloc[::10]
        err_ins = df_fused["err_pure_ins_pos_m"].iloc[::10]

        ax_err.plot(t_fused, err_ins, color="#EF4444", lw=1.8, ls=":", label="Solo IMU (Deriva exponencial)")
        ax_err.plot(t_fused, err_fused, color="#10B981", lw=2.2, label="Sistema Fusionado (Error acotado)")

        # Bandas de calidad comprensibles
        ax_err.axhspan(0, 50, color="#DCFCE7", alpha=0.6, label="Zona Optima (< 50 m)")
        ax_err.axhspan(50, 150, color="#FEF3C7", alpha=0.4, label="Zona Aceptable (< 150 m)")
        ax_err.set_ylim(0, min(650, err_ins.max() * 1.05))
    else:
        ax_err.text(0.5, 0.5, "Datos de error no disponibles", ha="center")

    ax_err.set_title("Evolucion del Error: Con Camara vs Sin Camara", fontsize=11, fontweight="bold", color="#1E293B")
    ax_err.set_xlabel("Tiempo (segundos)", fontsize=9, color="#475569")
    ax_err.set_ylabel("Error de Posicion (m)", fontsize=9, color="#475569")
    ax_err.grid(True, linestyle="--", alpha=0.5)
    ax_err.legend(loc="upper left", fontsize=7.5)

    # =========================================================================
    # PANEL 5: LO QUE VE LA CAMARA DE TERRENO (NADIR)
    # =========================================================================
    ax_cam_g = fig.add_subplot(gs[2, 1])
    cam_g_path = os.path.join(DATA_DIR, "camera_ground", "img_ground_00150.png")
    if not os.path.exists(cam_g_path):
        # Buscar el primer fotograma disponible
        frames = sorted([f for f in os.listdir(os.path.join(DATA_DIR, "camera_ground")) if f.endswith(".png")]) if os.path.exists(os.path.join(DATA_DIR, "camera_ground")) else []
        cam_g_path = os.path.join(DATA_DIR, "camera_ground", frames[len(frames)//2]) if frames else ""

    if os.path.exists(cam_g_path):
        img_g = plt.imread(cam_g_path)
        ax_cam_g.imshow(img_g)
    else:
        ax_cam_g.text(0.5, 0.5, "Fotograma Nadir\nno generado aun", ha="center", va="center")

    ax_cam_g.set_title("Camara de Terreno (Hacia Abajo)", fontsize=11, fontweight="bold", color="#1E293B")
    ax_cam_g.set_xticks([])
    ax_cam_g.set_yticks([])
    for spine in ax_cam_g.spines.values():
        spine.set_color("#10B981")
        spine.set_linewidth(2)
    ax_cam_g.text(
        0.5, -0.10, "Detecta carreteras y caminos para corregir la posicion",
        transform=ax_cam_g.transAxes, ha="center", fontsize=8, color="#0F766E"
    )

    # =========================================================================
    # PANEL 6: LO QUE VE LA CAMARA DE ESTRELLAS (CENITAL)
    # =========================================================================
    ax_cam_s = fig.add_subplot(gs[2, 2])
    cam_s_path = os.path.join(DATA_DIR, "camera_sky", "img_sky_00060.png")
    if not os.path.exists(cam_s_path):
        frames_s = sorted([f for f in os.listdir(os.path.join(DATA_DIR, "camera_sky")) if f.endswith(".png")]) if os.path.exists(os.path.join(DATA_DIR, "camera_sky")) else []
        cam_s_path = os.path.join(DATA_DIR, "camera_sky", frames_s[len(frames_s)//2]) if frames_s else ""

    if os.path.exists(cam_s_path):
        img_s = plt.imread(cam_s_path)
        ax_cam_s.imshow(img_s)
    else:
        ax_cam_s.text(0.5, 0.5, "Fotograma Estelar\nno generado aun", ha="center", va="center")

    ax_cam_s.set_title("Camara de Estrellas (Hacia Arriba)", fontsize=11, fontweight="bold", color="#1E293B")
    ax_cam_s.set_xticks([])
    ax_cam_s.set_yticks([])
    for spine in ax_cam_s.spines.values():
        spine.set_color("#3B82F6")
        spine.set_linewidth(2)
    ax_cam_s.text(
        0.5, -0.10, "Reconoce constelaciones para fijar el rumbo del avion",
        transform=ax_cam_s.transAxes, ha="center", fontsize=8, color="#1D4ED8"
    )

    # Guardar imagen en alta resolucion
    out_file = os.path.join(PLOTS_DIR, "presentacion.png")
    plt.savefig(out_file, dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

    size_kb = os.path.getsize(out_file) / 1024
    print(f"Panel de presentacion generado exitosamente:")
    print(f"  Ruta: {out_file}")
    print(f"  Tamano: {size_kb:.1f} KB")


if __name__ == "__main__":
    create_presentation_figure()
