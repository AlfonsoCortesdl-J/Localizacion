"""
visual_navigation.py
====================
Algoritmo de navegación aérea por visión artificial:
1. Odometría Visual (Visual Odometry, VO):
   - Estimación de velocidad y desplazamiento relativo cuadro a cuadro (Frame-to-Frame)
     utilizando descriptores ORB, emparejamiento k-NN con ratio test de Lowe y RANSAC.
2. Localización Relativa al Terreno (Map Matching / TRN):
   - Corrección periódica de deriva de posición mediante emparejamiento global con el
     ortomapa georreferenciado de referencia.
3. Evaluación de Error contra Ground Truth:
   - Cálculo de RMSE, Error Circular Probable (CEP-50 y CEP-95) y error de rumbo.
   - Generación de gráficas de evaluación comparativa.

Archivos generados:
-------------------
1. data/visual_navigation_results.csv: Estimaciones y errores paso a paso.
2. plots/visual_navigation_trajectory.png: Comparativa de trayectorias (GT vs VO vs Fusión).
3. plots/visual_navigation_errors.png: Perfil temporal de errores de posición y rumbo.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class VisualOdometryEstimator:
    """
    Estimador de Odometría Visual para cámaras cenitales/nadir en vehículos aéreos.
    """

    def __init__(self, fov_deg: float = 70.0, img_w: int = 640, img_h: int = 480):
        self.img_w = img_w
        self.img_h = img_h
        self.fov_deg = fov_deg
        self.focal_px = (self.img_w / 2.0) / np.tan(np.radians(self.fov_deg / 2.0))

        # Detector de características ORB optimizado para velocidad y robustez
        self.orb = cv2.ORB_create(
            nfeatures=600,
            scaleFactor=1.2,
            nlevels=4,
            edgeThreshold=15,
            fastThreshold=20
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def estimate_motion(
        self,
        img_prev: np.ndarray,
        img_curr: np.ndarray,
        alt_agl_m: float
    ) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
        """
        Estima el desplazamiento métrico (dx_body, dy_body) en metros y el cambio
        de ángulo de rumbo dyaw en radianes entre dos imágenes consecutivas.
        """
        gray1 = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY) if len(img_prev.shape) == 3 else img_prev
        gray2 = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY) if len(img_curr.shape) == 3 else img_curr

        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        kp2, des2 = self.orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None, None, None, 0

        # Emparejamiento con prueba de razón de Lowe
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.78 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 8:
            return None, None, None, len(good_matches)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimar transformación afín 2D rígida (Rotación + Traslación + Escala) con RANSAC
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)

        if M is None or inliers is None:
            return None, None, None, 0

        num_inliers = int(np.sum(inliers))
        if num_inliers < 6:
            return None, None, None, num_inliers

        # Descomposición de la matriz afín:
        # M = [ [s*cos(theta), -s*sin(theta), tx],
        #       [s*sin(theta),  s*cos(theta), ty] ]
        s_cos = M[0, 0]
        s_sin = M[1, 0]
        tx_px = M[0, 2]
        ty_px = M[1, 2]

        d_yaw_rad = np.arctan2(s_sin, s_cos)

        # Factor de escala pixel-a-metro (GSD en metros/píxel)
        gsd = alt_agl_m / self.focal_px

        # Conversión de píxeles a metros en ejes cuerpo de la aeronave:
        # En la convención Nadir: eje X_c (horizontal) es Y_cuerpo (ala derecha)
        # Eje Y_c (vertical) es -X_cuerpo (hacia la cola)
        dx_body_m = -ty_px * gsd  # Avance longitudinal (Forward)
        dy_body_m = tx_px * gsd   # Desplazamiento lateral (Right)

        return dx_body_m, dy_body_m, d_yaw_rad, num_inliers


class MapMatcher:
    """
    Localizador Absoluto basado en emparejamiento contra el Ortomapa de Referencia (TRN).
    """

    def __init__(self, map_path: str, meta_path: str):
        self.map_img = cv2.imread(map_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.gsd = self.meta["gsd_m_per_px"]
        self.center_lat = self.meta["center_geodetic"]["latitude_deg"]
        self.center_lon = self.meta["center_geodetic"]["longitude_deg"]
        self.meters_per_deg_lat = 111320.0
        self.meters_per_deg_lon = self.meters_per_deg_lat * np.cos(np.radians(self.center_lat))

        # Extractor de descriptores del mapa global
        self.orb = cv2.ORB_create(nfeatures=2500, scaleFactor=1.2, nlevels=5)
        gray_map = cv2.cvtColor(self.map_img, cv2.COLOR_BGR2GRAY)
        print("Indexando características visuales del ortomapa de referencia...")
        self.map_kp, self.map_des = self.orb.detectAndCompute(gray_map, None)
        print(f"  [OK] {len(self.map_kp)} puntos clave indexados en el mapa global.")
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match_frame(
        self,
        frame: np.ndarray,
        expected_pos_m: Tuple[float, float],
        search_radius_m: float = 350.0
    ) -> Optional[Tuple[float, float, float]]:
        """
        Empareja el fotograma actual contra el mapa global cerca de la posición esperada.
        Retorna (x_ned_m, y_ned_m, yaw_deg) estimados, o None si no hay certeza suficiente.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_kp, frame_des = self.orb.detectAndCompute(gray_frame, None)

        if frame_des is None or len(frame_kp) < 15:
            return None

        # Filtrar puntos del mapa dentro de la ventana de búsqueda de incertidumbre
        exp_e_m, exp_n_m = expected_pos_m
        exp_x_px = self.meta["width_px"] / 2.0 + (exp_e_m / self.gsd)
        exp_y_px = self.meta["height_px"] / 2.0 - (exp_n_m / self.gsd)
        radius_px = search_radius_m / self.gsd

        roi_indices = [
            i for i, kp in enumerate(self.map_kp)
            if (kp.pt[0] - exp_x_px) ** 2 + (kp.pt[1] - exp_y_px) ** 2 <= radius_px ** 2
        ]

        if len(roi_indices) < 20:
            return None

        sub_des = self.map_des[roi_indices]
        matches = self.matcher.knnMatch(frame_des, sub_des, k=2)

        good = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good.append((m.queryIdx, roi_indices[m.trainIdx]))

        if len(good) < 12:
            return None

        src_pts = np.float32([frame_kp[q].pt for q, t in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.map_kp[t].pt for q, t in good]).reshape(-1, 1, 2)

        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if M is None or inliers is None or np.sum(inliers) < 8:
            return None

        # Proyectar el centro óptico del fotograma hacia el mapa
        center_cam_pt = np.array([[[frame.shape[1] / 2.0, frame.shape[0] / 2.0]]], dtype=np.float32)
        center_map_pt = cv2.transform(center_cam_pt, M)[0, 0]

        # Convertir píxeles de mapa a coordenadas métricas NED (Este, Norte)
        est_e_m = (center_map_pt[0] - self.meta["width_px"] / 2.0) * self.gsd
        est_n_m = -(center_map_pt[1] - self.meta["height_px"] / 2.0) * self.gsd

        # Estimar rumbo a partir de la rotación afín (corrigiendo la inversión de eje Y de imagen)
        rot_rad = np.arctan2(M[1, 0], M[0, 0])
        est_yaw_deg = (360.0 - np.degrees(rot_rad)) % 360.0

        return est_e_m, est_n_m, est_yaw_deg


def run_visual_navigation(
    ground_csv_path: str = "data/camera_ground_frames.csv",
    frames_dir: str = "data/camera_ground",
    map_path: str = "data/reference_map.png",
    meta_path: str = "data/reference_map_meta.json",
    map_match_interval: int = 15  # Intentar Map Matching cada 15 fotogramas (~3 segundos)
) -> None:
    """Ejecuta el pipeline completo de navegación por visión y genera métricas de validación."""
    print(f"Cargando metadatos de vuelo desde: {ground_csv_path}...")
    df_meta = pd.read_csv(ground_csv_path)

    vo = VisualOdometryEstimator(fov_deg=70.0, img_w=640, img_h=480)
    matcher = MapMatcher(map_path, meta_path)

    # 1. Coordenadas reales Ground Truth respecto al centro del mapa
    center_lat = matcher.center_lat
    center_lon = matcher.center_lon
    meters_per_deg_lat = matcher.meters_per_deg_lat
    meters_per_deg_lon = matcher.meters_per_deg_lon

    df_meta["gt_east_m"] = (df_meta["lon_geod_deg"] - center_lon) * meters_per_deg_lon
    df_meta["gt_north_m"] = (df_meta["lat_geod_deg"] - center_lat) * meters_per_deg_lat

    # Vectores de estado estimados
    # Estado inicial: Inicializado con la posición real de despegue/inicio (t=0)
    est_vo_east = [df_meta["gt_east_m"].iloc[0]]
    est_vo_north = [df_meta["gt_north_m"].iloc[0]]
    est_vo_yaw = [df_meta["yaw_deg"].iloc[0]]

    est_fused_east = [df_meta["gt_east_m"].iloc[0]]
    est_fused_north = [df_meta["gt_north_m"].iloc[0]]
    est_fused_yaw = [df_meta["yaw_deg"].iloc[0]]

    map_corrections = []  # Puntos donde actuó el Map Matching

    print(f"\nIniciando estimación de Odometría Visual y Map Matching sobre {len(df_meta)} fotogramas...")

    prev_img_path = os.path.join(frames_dir, df_meta.iloc[0]["image_filename"])
    prev_img = cv2.imread(prev_img_path)

    for i in range(1, len(df_meta)):
        row = df_meta.iloc[i]
        curr_img_path = os.path.join(frames_dir, row["image_filename"])
        curr_img = cv2.imread(curr_img_path)

        alt_agl = float(row["altitude_agl_m"])

        # A. Paso de Odometría Visual (Frame-to-Frame)
        dx_b, dy_b, dyaw, inliers = vo.estimate_motion(prev_img, curr_img, alt_agl)

        if dx_b is None:
            # En caso de pérdida temporal de puntos, mantener velocidad previa (dead reckoning)
            dx_b = 0.0
            dy_b = 0.0
            dyaw = 0.0

        # Integrar odometría pura (sin corrección de mapa)
        current_vo_yaw_deg = (est_vo_yaw[-1] + np.degrees(dyaw)) % 360.0
        yaw_rad = np.radians(current_vo_yaw_deg)

        # Rotar desplazamiento de ejes cuerpo a marco NED
        d_north_vo = dx_b * np.cos(yaw_rad) - dy_b * np.sin(yaw_rad)
        d_east_vo = dx_b * np.sin(yaw_rad) + dy_b * np.cos(yaw_rad)

        new_vo_e = est_vo_east[-1] + d_east_vo
        new_vo_n = est_vo_north[-1] + d_north_vo

        est_vo_east.append(new_vo_e)
        est_vo_north.append(new_vo_n)
        est_vo_yaw.append(current_vo_yaw_deg)

        # B. Integrar en estimador fusionado con corrección de mapa (TRN)
        current_fused_yaw_deg = (est_fused_yaw[-1] + np.degrees(dyaw)) % 360.0
        fused_yaw_rad = np.radians(current_fused_yaw_deg)

        d_north_fused = dx_b * np.cos(fused_yaw_rad) - dy_b * np.sin(fused_yaw_rad)
        d_east_fused = dx_b * np.sin(fused_yaw_rad) + dy_b * np.cos(fused_yaw_rad)

        new_fused_e = est_fused_east[-1] + d_east_fused
        new_fused_n = est_fused_north[-1] + d_north_fused

        # C. Intentar corrección de mapa periódica (Map Matching)
        map_match_success = False
        if i % map_match_interval == 0:
            match_res = matcher.match_frame(curr_img, (new_fused_e, new_fused_n), search_radius_m=350.0)
            if match_res is not None:
                map_e, map_n, map_yaw = match_res
                # Filtro de actualización suave (ponderación tipo Kalman / Complementario)
                alpha = 0.85  # Confianza alta en el mapa georreferenciado
                new_fused_e = (1 - alpha) * new_fused_e + alpha * map_e
                new_fused_n = (1 - alpha) * new_fused_n + alpha * map_n
                diff_yaw = (map_yaw - current_fused_yaw_deg + 180.0) % 360.0 - 180.0
                current_fused_yaw_deg = (current_fused_yaw_deg + alpha * diff_yaw) % 360.0
                map_corrections.append((i, row["sim_time_sec"], new_fused_e, new_fused_n))
                map_match_success = True

        est_fused_east.append(new_fused_e)
        est_fused_north.append(new_fused_n)
        est_fused_yaw.append(current_fused_yaw_deg)

        prev_img = curr_img

        if i % 50 == 0 or i == len(df_meta) - 1:
            err_vo = np.sqrt((new_vo_e - row["gt_east_m"])**2 + (new_vo_n - row["gt_north_m"])**2)
            err_fused = np.sqrt((new_fused_e - row["gt_east_m"])**2 + (new_fused_n - row["gt_north_m"])**2)
            print(f"  Frame {i:03d} (t={row['sim_time_sec']:5.1f}s) | Error VO pura: {err_vo:6.2f} m | Error VO+Mapa: {err_fused:6.2f} m | Inliers: {inliers:3d}")

    # 2. Agregar resultados y calcular métricas formales de navegación
    df_res = df_meta.copy()
    df_res["est_vo_east_m"] = est_vo_east
    df_res["est_vo_north_m"] = est_vo_north
    df_res["est_vo_yaw_deg"] = est_vo_yaw

    df_res["est_fused_east_m"] = est_fused_east
    df_res["est_fused_north_m"] = est_fused_north
    df_res["est_fused_yaw_deg"] = est_fused_yaw

    # Errores en metros
    df_res["error_vo_m"] = np.sqrt((df_res["est_vo_east_m"] - df_res["gt_east_m"])**2 +
                                   (df_res["est_vo_north_m"] - df_res["gt_north_m"])**2)
    df_res["error_fused_m"] = np.sqrt((df_res["est_fused_east_m"] - df_res["gt_east_m"])**2 +
                                      (df_res["est_fused_north_m"] - df_res["gt_north_m"])**2)

    # Errores angulares de rumbo en grados
    df_res["error_yaw_deg"] = np.abs((df_res["est_vo_yaw_deg"] - df_res["yaw_deg"] + 180.0) % 360.0 - 180.0)

    # Guardar CSV de resultados
    data_dir = os.path.join(parent_dir, "data")
    out_csv = os.path.join(data_dir, "visual_navigation_results.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\n[Archivo guardado] Resultados exportados en: {out_csv}")

    # Calcular estadísticas de rendimiento (RMSE, CEP-50, CEP-95)
    rmse_vo = np.sqrt(np.mean(df_res["error_vo_m"]**2))
    rmse_fused = np.sqrt(np.mean(df_res["error_fused_m"]**2))
    cep50_fused = np.percentile(df_res["error_fused_m"], 50)
    cep95_fused = np.percentile(df_res["error_fused_m"], 95)
    max_err_fused = np.max(df_res["error_fused_m"])

    print("\n" + "="*60)
    print("MÉTRICAS FORMALES DE EVALUACIÓN DE NAVEGACIÓN POR VISIÓN")
    print("="*60)
    print(f"• Distancia total recorrida por la aeronave : {df_meta['gt_east_m'].iloc[-1] - df_meta['gt_east_m'].iloc[0]:.1f} m Este")
    print(f"• Error RMSE - Odometría Visual pura (con deriva) : {rmse_vo:.2f} m")
    print(f"• Error RMSE - Odometría + Corrección de Mapa (TRN): {rmse_fused:.2f} m")
    print(f"• Error Circular Probable 50% (CEP-50)            : {cep50_fused:.2f} m")
    print(f"• Error Circular Probable 95% (CEP-95)            : {cep95_fused:.2f} m")
    print(f"• Error Máximo de Posición (VO + Mapa)           : {max_err_fused:.2f} m")
    print(f"• Número de correcciones exitosas de mapa         : {len(map_corrections)}")
    print("="*60)

    # 3. Generar Gráficas de Análisis
    plots_dir = os.path.join(parent_dir, "plots")
    plot_visual_nav_trajectory(df_res, map_corrections, matcher, plots_dir)
    plot_visual_nav_errors(df_res, rmse_vo, rmse_fused, plots_dir)


def plot_visual_nav_trajectory(
    df: pd.DataFrame,
    map_corrections: list,
    matcher: MapMatcher,
    plots_dir: str
) -> str:
    """Genera la figura comparativa de trayectorias sobre el ortomapa."""
    fig, ax = plt.subplots(figsize=(11, 9))

    # Convertir mapa BGR a RGB
    map_rgb = cv2.cvtColor(matcher.map_img, cv2.COLOR_BGR2RGB)
    half_span_m = matcher.meta["width_m"] / 2.0
    extent = [-half_span_m, half_span_m, -half_span_m, half_span_m]

    ax.imshow(map_rgb, extent=extent, origin="upper", alpha=0.88)

    # Trayectoria Ground Truth (Real)
    ax.plot(df["gt_east_m"], df["gt_north_m"], color="#2ca02c", linewidth=3.0, label="Ground Truth (Simulador Real)", zorder=4)

    # Trayectoria Odometría Visual Pura (VO)
    ax.plot(df["est_vo_east_m"], df["est_vo_north_m"], color="#d62728", linewidth=2.0, linestyle="--", label="Odometría Visual Pura (Deriva acumulativa)", zorder=5)

    # Trayectoria Fusionada (VO + Map Matching)
    ax.plot(df["est_fused_east_m"], df["est_fused_north_m"], color="#1f77b4", linewidth=2.5, linestyle="-", label="Navegación por Visión (VO + Map Matching)", zorder=6)

    # Marcadores de corrección de mapa
    if map_corrections:
        corr_e = [c[2] for c in map_corrections]
        corr_n = [c[3] for c in map_corrections]
        ax.scatter(corr_e, corr_n, color="#ff7f0e", marker="^", s=70, edgecolor="black", label=f"Correcciones de Mapa TRN (n={len(map_corrections)})", zorder=7)

    # Marcadores de Inicio y Fin
    ax.plot(df["gt_east_m"].iloc[0], df["gt_north_m"].iloc[0], "o", color="#2ca02c", markersize=10, markeredgecolor="white", markeredgewidth=2, label="Inicio de Vuelo", zorder=8)
    ax.plot(df["gt_east_m"].iloc[-1], df["gt_north_m"].iloc[-1], "s", color="#d62728", markersize=10, markeredgecolor="white", markeredgewidth=2, label="Fin de Vuelo", zorder=8)

    ax.set_title("Navegación Aérea por Visión: Ground Truth vs Odometría vs Corrección TRN", fontsize=12, fontweight="bold")
    ax.set_xlabel("Distancia Este NED (metros)", fontsize=11)
    ax.set_ylabel("Distancia Norte NED (metros)", fontsize=11)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, facecolor="white")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(plots_dir, "visual_navigation_trajectory.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  [Gráfico generado] {out_path}")
    return out_path


def plot_visual_nav_errors(
    df: pd.DataFrame,
    rmse_vo: float,
    rmse_fused: float,
    plots_dir: str
) -> str:
    """Genera el gráfico del error de posición instantáneo y error de rumbo."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    t = df["sim_time_sec"].values

    # 1. Error de Posición Horizontal en metros
    ax1.plot(t, df["error_vo_m"].values, color="#d62728", linewidth=2.0, label=f"Error VO Pura (RMSE = {rmse_vo:.1f} m)")
    ax1.plot(t, df["error_fused_m"].values, color="#1f77b4", linewidth=2.2, label=f"Error VO + Map Matching (RMSE = {rmse_fused:.1f} m)")
    ax1.axhline(rmse_fused, color="#1f77b4", linestyle=":", linewidth=1.5, label="Límite RMSE Fusión")
    ax1.set_ylabel("Error de Posición (m)", fontsize=11, fontweight="bold")
    ax1.set_title("Evolución del Error de Estimación de Posición Horizontal y Rumbo", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Error de Orientación / Rumbo en grados
    ax2.plot(t, df["error_yaw_deg"].values, color="#ff7f0e", linewidth=1.8, label="Error de Rumbo |ψ_est - ψ_gt|")
    ax2.set_ylabel("Error de Rumbo (°)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Tiempo de simulación (s)", fontsize=11)
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(plots_dir, "visual_navigation_errors.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  [Gráfico generado] {out_path}")
    return out_path


if __name__ == "__main__":
    csv_file = os.path.join(parent_dir, "data", "camera_ground_frames.csv")
    frames_path = os.path.join(parent_dir, "data", "camera_ground")
    map_file = os.path.join(parent_dir, "data", "reference_map.png")
    meta_file = os.path.join(parent_dir, "data", "reference_map_meta.json")
    run_visual_navigation(csv_file, frames_path, map_file, meta_file)
