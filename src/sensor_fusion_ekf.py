"""
sensor_fusion_ekf.py
====================
Filtro de Kalman Extendido (EKF) para fusión sensorial inercial y visual:
- Predicción a alta frecuencia (100 Hz):
  Mecanización inercial (INS) con lecturas de acelerómetro triaxial (fuerza específica)
  y giróscopo triaxial (velocidades angulares).
- Corrección a baja frecuencia (5 Hz):
  Mediciones de posición y rumbo provenientes del algoritmo de navegación por visión (VO + TRN).

Vector de Estado del EKF (9 variables):
---------------------------------------
x = [ p_North, p_East, p_Down, v_North, v_East, v_Down, roll, pitch, yaw ]^T

Métricas evaluadas:
-------------------
- Comparación directa contra Ground Truth del simulador JSBSim.
- Envolventes de incertidumbre 3-sigma derivadas de la matriz de covarianza P.
- Cálculo de RMSE y reducción de varianza.

Archivos generados:
-------------------
1. data/fused_navigation_log.csv: Estado estimado completo a 100 Hz y cotas 3-sigma.
2. plots/sensor_fusion_trajectory.png: Trayectoria GT vs IMU Pura vs Visión vs Fusión EKF.
3. plots/sensor_fusion_errors.png: Perfil de errores de posición y actitud con cotas 3-sigma.
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


class AerialEKFFusion:
    """
    Filtro de Kalman Extendido para navegación integrada INS/Visual.
    """

    GRAVITY_MPS2 = 9.80665

    def __init__(self, init_pos_ned: np.ndarray, init_vel_ned: np.ndarray, init_euler_deg: np.ndarray):
        # Estado: [pN, pE, pD, vN, vE, vD, phi, theta, psi]
        phi_rad = np.radians(init_euler_deg[0])
        theta_rad = np.radians(init_euler_deg[1])
        psi_rad = np.radians(init_euler_deg[2])

        self.x = np.array([
            init_pos_ned[0], init_pos_ned[1], init_pos_ned[2],
            init_vel_ned[0], init_vel_ned[1], init_vel_ned[2],
            phi_rad, theta_rad, psi_rad
        ], dtype=np.float64)

        # Matriz de covarianza de error P (9x9)
        self.P = np.diag([
            5.0**2, 5.0**2, 2.0**2,       # Incertidumbre inicial de posición (m^2)
            1.0**2, 1.0**2, 0.5**2,       # Incertidumbre inicial de velocidad (m^2/s^2)
            np.radians(2.0)**2, np.radians(2.0)**2, np.radians(3.0)**2  # Incertidumbre de actitud (rad^2)
        ])

        # Ruido de proceso Q (incertidumbre de sensores inerciales MEMS)
        # Ruido continuo de acelerómetro y giróscopo
        self.accel_noise_var = (0.25)**2     # (m/s^2)^2
        self.gyro_noise_var = (np.radians(0.8))**2  # (rad/s)^2

        # Ruido de medición R (incertidumbre de la visión)
        # R_vis = [var_pN, var_pE, var_psi]
        self.R_vis = np.diag([
            12.0**2, 12.0**2, np.radians(2.5)**2  # Visión: ~12m en posición, ~2.5 deg en rumbo
        ])

    def rotation_matrix_body_to_ned(self, phi: float, theta: float, psi: float) -> np.ndarray:
        """Matriz de rotación R_b^ned a partir de ángulos de Euler Z-Y-X."""
        c_ph, s_ph = np.cos(phi), np.sin(phi)
        c_th, s_th = np.cos(theta), np.sin(theta)
        c_ps, s_ps = np.cos(psi), np.sin(psi)

        R = np.array([
            [c_th * c_ps, s_ph * s_th * c_ps - c_ph * s_ps, c_ph * s_th * c_ps + s_ph * s_ps],
            [c_th * s_ps, s_ph * s_th * s_ps + c_ph * c_ps, c_ph * s_th * s_ps - s_ph * c_ps],
            [-s_th,       s_ph * c_th,                     c_ph * c_th]
        ], dtype=np.float64)
        return R

    @staticmethod
    def euler_rate_matrix(phi: float, theta: float) -> np.ndarray:
        """Matriz W tal que [dphi, dtheta, dpsi] = W @ [p, q, r] (Euler Z-Y-X)."""
        th = np.clip(theta, -np.radians(85.0), np.radians(85.0))
        c_ph, s_ph = np.cos(phi), np.sin(phi)
        return np.array([
            [1.0, s_ph * np.tan(th), c_ph * np.tan(th)],
            [0.0, c_ph, -s_ph],
            [0.0, s_ph / np.cos(th), c_ph / np.cos(th)]
        ], dtype=np.float64)

    def predict(self, accel_body_mps2: np.ndarray, gyro_body_rad_s: np.ndarray, dt: float) -> None:
        """
        Paso de predicción EKF (Mecanización Inercial INS a 100 Hz).
        """
        phi = self.x[6]
        theta = self.x[7]
        psi = self.x[8]

        # 1. Integración de Actitud (Cinemática angular de Euler)
        # [dphi, dtheta, dpsi]^T = W * [p, q, r]^T
        c_ph, s_ph = np.cos(phi), np.sin(phi)
        t_th = np.tan(np.clip(theta, -np.radians(85.0), np.radians(85.0)))
        sec_th = 1.0 / np.cos(np.clip(theta, -np.radians(85.0), np.radians(85.0)))

        W = np.array([
            [1.0, s_ph * t_th, c_ph * t_th],
            [0.0, c_ph, -s_ph],
            [0.0, s_ph * sec_th, c_ph * sec_th]
        ], dtype=np.float64)

        euler_rates = W @ gyro_body_rad_s
        phi_new = phi + euler_rates[0] * dt
        theta_new = theta + euler_rates[1] * dt
        psi_new = (psi + euler_rates[2] * dt) % (2 * np.pi)

        # 2. Integración de Velocidad y Posición (Transformación de fuerza específica)
        R_b_ned = self.rotation_matrix_body_to_ned(phi, theta, psi)
        f_ned = R_b_ned @ accel_body_mps2

        # Aceleración inercial = Fuerza específica + Gravedad NED
        a_ned = f_ned + np.array([0.0, 0.0, self.GRAVITY_MPS2])

        vel = self.x[3:6]
        pos = self.x[0:3]

        pos_new = pos + vel * dt + 0.5 * a_ned * (dt**2)
        vel_new = vel + a_ned * dt

        self.x[0:3] = pos_new
        self.x[3:6] = vel_new
        self.x[6] = phi_new
        self.x[7] = theta_new
        self.x[8] = psi_new

        # 3. Propagación de Covarianza P_k = F * P * F^T + Q
        F = np.eye(9, dtype=np.float64)
        F[0:3, 3:6] = np.eye(3) * dt
        # Acoplamiento de aceleración respecto a orientación en F[3:6, 6:9]
        # CORRECCION: el estado de actitud son angulos de Euler, asi que los Jacobianos
        # se calculan respecto a (phi, theta, psi). Antes: -R*skew(f) (modelo de vector de
        # rotacion) y "+= W*dt" (W multiplica al giroscopo, no a los angulos: hacia crecer
        # la covarianza de actitud exponencialmente).
        eps = 1e-6
        ang = np.array([phi, theta, psi])
        J_acc = np.zeros((3, 3))
        J_att = np.zeros((3, 3))
        for i in range(3):
            d = np.zeros(3)
            d[i] = eps
            a_p, a_m = ang + d, ang - d
            J_acc[:, i] = (self.rotation_matrix_body_to_ned(*a_p) @ accel_body_mps2
                           - self.rotation_matrix_body_to_ned(*a_m) @ accel_body_mps2) / (2 * eps)
            J_att[:, i] = (self.euler_rate_matrix(a_p[0], a_p[1]) @ gyro_body_rad_s
                           - self.euler_rate_matrix(a_m[0], a_m[1]) @ gyro_body_rad_s) / (2 * eps)
        F[3:6, 6:9] = J_acc * dt
        F[6:9, 6:9] += J_att * dt

        Q = np.zeros((9, 9), dtype=np.float64)
        Q[3:6, 3:6] = np.eye(3) * self.accel_noise_var * dt
        Q[6:9, 6:9] = np.eye(3) * self.gyro_noise_var * dt

        self.P = F @ self.P @ F.T + Q

    def update_vision(self, p_north_vis: float, p_east_vis: float, psi_vis_rad: float) -> None:
        """
        Paso de corrección EKF con actualización visual (5 Hz).
        """
        # Matriz de observación H (3x9): Medimos pN, pE, psi
        H = np.zeros((3, 9), dtype=np.float64)
        H[0, 0] = 1.0  # p_North
        H[1, 1] = 1.0  # p_East
        H[2, 8] = 1.0  # psi

        z = np.array([p_north_vis, p_east_vis, psi_vis_rad], dtype=np.float64)
        h_x = np.array([self.x[0], self.x[1], self.x[8]], dtype=np.float64)

        # Innovación / Residual con normalización de ángulo de rumbo
        y = z - h_x
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi

        # Covarianza de innovación S = H * P * H^T + R
        S = H @ self.P @ H.T + self.R_vis
        K = self.P @ H.T @ np.linalg.inv(S)

        # Actualización de estado
        self.x += K @ y
        self.x[8] = self.x[8] % (2 * np.pi)

        # Actualización de covarianza en forma Joseph para estabilidad numérica
        I_KH = np.eye(9) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_vis @ K.T

    def _skew(self, v: np.ndarray) -> np.ndarray:
        """Matriz antisimétrica (cross-product matrix)."""
        return np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0]
        ], dtype=np.float64)


def run_sensor_fusion(
    imu_csv_path: str = "data/imu_measurements.csv",
    vis_csv_path: str = "data/visual_navigation_results.csv",
    flight_log_path: str = "data/flight_log.csv",
    meta_json_path: str = "data/reference_map_meta.json"
) -> None:
    """Ejecuta el pipeline completo de fusión EKF y genera análisis comparativo."""
    print("Cargando conjuntos de datos para fusión sensorial...")
    df_imu = pd.read_csv(imu_csv_path)
    df_vis = pd.read_csv(vis_csv_path)
    df_gt = pd.read_csv(flight_log_path)

    with open(meta_json_path, "r", encoding="utf-8") as f:
        map_meta = json.load(f)

    center_lat = map_meta["center_geodetic"]["latitude_deg"]
    center_lon = map_meta["center_geodetic"]["longitude_deg"]
    m_lat = 111320.0
    m_lon = m_lat * np.cos(np.radians(center_lat))

    # Convertir Ground Truth a coordenadas métricas relativas (North, East)
    df_gt["gt_east_m"] = (df_gt["lon_geod_deg"] - center_lon) * m_lon
    df_gt["gt_north_m"] = (df_gt["lat_geod_deg"] - center_lat) * m_lat

    # Estado inicial real del vuelo
    init_p_ned = np.array([df_gt["gt_north_m"].iloc[0], df_gt["gt_east_m"].iloc[0], -df_gt["altitude_msl_m"].iloc[0]])
    init_v_ned = np.array([df_gt["vel_north_mps"].iloc[0], df_gt["vel_east_mps"].iloc[0], df_gt["vel_down_mps"].iloc[0]])
    init_euler_deg = np.array([df_gt["roll_deg"].iloc[0], df_gt["pitch_deg"].iloc[0], df_gt["yaw_deg"].iloc[0]])

    ekf = AerialEKFFusion(init_p_ned, init_v_ned, init_euler_deg)

    # También mantenemos un integrador puramente inercial (sin visión) para ilustrar la deriva
    pure_ins_pos = init_p_ned.copy()
    pure_ins_vel = init_v_ned.copy()
    pure_ins_euler = np.radians(init_euler_deg.copy())

    # Emparejamiento temporal: IMU corre a 100 Hz (dt = 0.01s), Visión llega a 5 Hz
    vis_idx = 0
    num_vis = len(df_vis)

    history = []
    pure_ins_history = []

    t_imu = df_imu["sim_time_sec"].to_numpy(dtype=float)
    dt_med = float(np.median(np.diff(t_imu)))
    imu_hz = 1.0 / dt_med
    print(f"Ejecutando EKF INS/Visual sobre {len(df_imu)} muestras de IMU ({imu_hz:.1f} Hz, dt = {dt_med:.5f} s)...")
    t_prev = None

    for k in range(len(df_imu)):
        row_imu = df_imu.iloc[k]
        t = row_imu["sim_time_sec"]

        # CORRECCION: dt real entre muestras (antes fijo en 0.01 s aunque la IMU va a ~60 Hz,
        # lo que integraba solo ~60% del tiempo real y hacia que el INS "derivara" ~500 m).
        dt = dt_med if t_prev is None else (t - t_prev)
        t_prev = t
        accel = np.array([row_imu["accel_x_mps2"], row_imu["accel_y_mps2"], row_imu["accel_z_mps2"]])
        gyro = np.array([row_imu["gyro_p_rad_s"], row_imu["gyro_q_rad_s"], row_imu["gyro_r_rad_s"]])

        # 1. Predicción EKF
        ekf.predict(accel, gyro, dt)

        # 2. Mecanización INS Pura (sin correcciones para comparar divergencia)
        R_ins = ekf.rotation_matrix_body_to_ned(pure_ins_euler[0], pure_ins_euler[1], pure_ins_euler[2])
        a_ins_ned = R_ins @ accel + np.array([0.0, 0.0, 9.80665])
        pure_ins_pos += pure_ins_vel * dt + 0.5 * a_ins_ned * (dt**2)
        pure_ins_vel += a_ins_ned * dt
        # CORRECCION: la INS pura nunca actualizaba su actitud (usaba siempre la inicial).
        W_ins = ekf.euler_rate_matrix(pure_ins_euler[0], pure_ins_euler[1])
        pure_ins_euler = pure_ins_euler + (W_ins @ gyro) * dt
        pure_ins_history.append((pure_ins_pos[0], pure_ins_pos[1]))

        # 3. Corrección Visual periódica (cuando coincide el timestamp de visión)
        if vis_idx < num_vis and t >= df_vis.iloc[vis_idx]["sim_time_sec"] - 1e-4:
            row_vis = df_vis.iloc[vis_idx]
            p_n_vis = float(row_vis["est_fused_north_m"])
            p_e_vis = float(row_vis["est_fused_east_m"])
            psi_vis = np.radians(float(row_vis["est_fused_yaw_deg"]))
            ekf.update_vision(p_n_vis, p_e_vis, psi_vis)
            vis_idx += 1

        # Extraer cotas de incertidumbre 3-sigma
        sigma_pn = 3.0 * np.sqrt(max(0.0, ekf.P[0, 0]))
        sigma_pe = 3.0 * np.sqrt(max(0.0, ekf.P[1, 1]))
        sigma_psi_deg = 3.0 * np.degrees(np.sqrt(max(0.0, ekf.P[8, 8])))

        history.append({
            "sim_time_sec": t,
            "fused_north_m": ekf.x[0],
            "fused_east_m": ekf.x[1],
            "fused_down_m": ekf.x[2],
            "fused_vn_mps": ekf.x[3],
            "fused_ve_mps": ekf.x[4],
            "fused_vd_mps": ekf.x[5],
            "fused_roll_deg": np.degrees(ekf.x[6]),
            "fused_pitch_deg": np.degrees(ekf.x[7]),
            "fused_yaw_deg": np.degrees(ekf.x[8]) % 360.0,
            "sigma_3_north_m": sigma_pn,
            "sigma_3_east_m": sigma_pe,
            "sigma_3_yaw_deg": sigma_psi_deg,
        })

    df_fused = pd.DataFrame(history)
    df_fused["pure_ins_north_m"] = [p[0] for p in pure_ins_history]
    df_fused["pure_ins_east_m"] = [p[1] for p in pure_ins_history]

    # Interpolar Ground Truth a los instantes exactos de 100 Hz de la IMU para evaluación
    df_fused["gt_north_m"] = np.interp(df_fused["sim_time_sec"], df_gt["sim_time_sec"], df_gt["gt_north_m"])
    df_fused["gt_east_m"] = np.interp(df_fused["sim_time_sec"], df_gt["sim_time_sec"], df_gt["gt_east_m"])
    df_fused["gt_yaw_deg"] = np.interp(df_fused["sim_time_sec"], df_gt["sim_time_sec"], df_gt["yaw_deg"])
    df_fused["gt_vn_mps"] = np.interp(df_fused["sim_time_sec"], df_gt["sim_time_sec"], df_gt["vel_north_mps"])
    df_fused["gt_ve_mps"] = np.interp(df_fused["sim_time_sec"], df_gt["sim_time_sec"], df_gt["vel_east_mps"])

    # Errores en metros
    df_fused["err_fused_pos_m"] = np.sqrt(
        (df_fused["fused_east_m"] - df_fused["gt_east_m"])**2 +
        (df_fused["fused_north_m"] - df_fused["gt_north_m"])**2
    )
    df_fused["err_pure_ins_pos_m"] = np.sqrt(
        (df_fused["pure_ins_east_m"] - df_fused["gt_east_m"])**2 +
        (df_fused["pure_ins_north_m"] - df_fused["gt_north_m"])**2
    )
    df_fused["err_fused_yaw_deg"] = np.abs(
        (df_fused["fused_yaw_deg"] - df_fused["gt_yaw_deg"] + 180.0) % 360.0 - 180.0
    )

    data_dir = os.path.join(parent_dir, "data")
    fused_csv = os.path.join(data_dir, "fused_navigation_log.csv")
    df_fused.to_csv(fused_csv, index=False)
    print(f"\n[Archivo guardado] Log de navegación fusionada EKF en: {fused_csv}")

    # Métricas formales
    rmse_ins = np.sqrt(np.mean(df_fused["err_pure_ins_pos_m"]**2))
    rmse_fused = np.sqrt(np.mean(df_fused["err_fused_pos_m"]**2))
    cep50_fused = np.percentile(df_fused["err_fused_pos_m"], 50)
    cep95_fused = np.percentile(df_fused["err_fused_pos_m"], 95)
    mean_yaw_err = np.mean(df_fused["err_fused_yaw_deg"])

    print("\n" + "="*65)
    print("MÉTRICAS FORMALES DE FUSIÓN SENSORIAL EKF (INS + VISIÓN)")
    print("="*65)
    print(f"• Error RMSE - INS Pura sin correcciones (deriva inercial) : {rmse_ins:8.2f} m")
    print(f"• Error RMSE - Navegación Fusionada EKF (INS + Visión)     : {rmse_fused:8.2f} m")
    print(f"• Error Circular Probable 50% (CEP-50)                    : {cep50_fused:8.2f} m")
    print(f"• Error Circular Probable 95% (CEP-95)                    : {cep95_fused:8.2f} m")
    print(f"• Error medio de actitud / rumbo                          : {mean_yaw_err:8.2f}°")
    print(f"• Frecuencia de solución de navegación disponible         : {imu_hz:.1f} Hz (tasa de la IMU)")
    print("="*65)

    # Gráficas
    plots_dir = os.path.join(parent_dir, "plots")
    plot_fusion_trajectory(df_fused, df_vis, map_meta, plots_dir)
    plot_fusion_errors(df_fused, rmse_fused, plots_dir)


def plot_fusion_trajectory(df_fused: pd.DataFrame, df_vis: pd.DataFrame, map_meta: dict, plots_dir: str) -> str:
    """Genera gráfico comparativo de la trayectoria fusionada EKF vs Ground Truth e INS."""
    fig, ax = plt.subplots(figsize=(11, 9))

    half_span_m = map_meta["width_m"] / 2.0
    extent = [-half_span_m, half_span_m, -half_span_m, half_span_m]

    # Fondo de ortomapa
    map_img = cv2.imread(os.path.join(parent_dir, "data", "reference_map.png"))
    map_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
    ax.imshow(map_rgb, extent=extent, origin="upper", alpha=0.88)

    # 1. Ground Truth
    ax.plot(df_fused["gt_east_m"], df_fused["gt_north_m"], color="#2ca02c", linewidth=3.2, label="Ground Truth (Simulador Real)", zorder=4)

    # 2. INS Pura (Deriva acelerada)
    ax.plot(df_fused["pure_ins_east_m"], df_fused["pure_ins_north_m"], color="#7f7f7f", linewidth=1.5, linestyle=":", label="INS Pura (Deriva cuadrática inercial)", zorder=3)

    # 3. Visión Pura (Discreta a 5 Hz)
    ax.scatter(df_vis["est_fused_east_m"], df_vis["est_fused_north_m"], color="#ff7f0e", s=18, alpha=0.7, label="Mediciones de Visión (5 Hz)", zorder=5)

    # 4. Solución Fusionada EKF (Continua a 100 Hz)
    ax.plot(df_fused["fused_east_m"], df_fused["fused_north_m"], color="#1f77b4", linewidth=2.4, label="Navegación Fusionada EKF (100 Hz)", zorder=6)

    # Marcadores de Inicio y Fin
    ax.plot(df_fused["gt_east_m"].iloc[0], df_fused["gt_north_m"].iloc[0], "o", color="#2ca02c", markersize=10, markeredgecolor="white", markeredgewidth=2, label="Inicio", zorder=7)
    ax.plot(df_fused["gt_east_m"].iloc[-1], df_fused["gt_north_m"].iloc[-1], "s", color="#d62728", markersize=10, markeredgecolor="white", markeredgewidth=2, label="Fin", zorder=7)

    ax.set_title("Fusión Sensorial EKF: Inercial (100 Hz) + Visión Artificial (5 Hz)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Distancia Este NED (metros)", fontsize=11)
    ax.set_ylabel("Distancia Norte NED (metros)", fontsize=11)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, facecolor="white")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(plots_dir, "sensor_fusion_trajectory.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  [Gráfico generado] {out_path}")
    return out_path


def plot_fusion_errors(df_fused: pd.DataFrame, rmse_fused: float, plots_dir: str) -> str:
    """Genera gráfico de evolución temporal del error con envolventes 3-sigma."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    t = df_fused["sim_time_sec"].values

    # 1. Error de Posición Horizontal vs Cotas de Covarianza 3-Sigma
    err_pos = df_fused["err_fused_pos_m"].values
    sigma_pos = np.sqrt(df_fused["sigma_3_north_m"]**2 + df_fused["sigma_3_east_m"]**2)

    ax1.plot(t, err_pos, color="#1f77b4", linewidth=1.8, label=f"Error de Posición Horizontal (RMSE = {rmse_fused:.2f} m)")
    ax1.plot(t, sigma_pos, color="#d62728", linestyle="--", linewidth=1.5, label="Envolvente de Incertidumbre 3σ")
    ax1.fill_between(t, 0, sigma_pos, color="#d62728", alpha=0.08)
    ax1.set_ylabel("Error Posición (m)", fontsize=10, fontweight="bold")
    ax1.set_title("Desempeño del Filtro de Kalman Extendido (INS/Visión) con Límites 3-Sigma", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Error de Velocidad en Marco NED
    err_vn = np.abs(df_fused["fused_vn_mps"].values - df_fused["gt_vn_mps"].values)
    err_ve = np.abs(df_fused["fused_ve_mps"].values - df_fused["gt_ve_mps"].values)
    ax2.plot(t, err_vn, color="#2ca02c", linewidth=1.5, label="|V_North error| (m/s)")
    ax2.plot(t, err_ve, color="#ff7f0e", linewidth=1.5, label="|V_East error| (m/s)")
    ax2.set_ylabel("Error Velocidad (m/s)", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Error de Rumbo / Actitud
    err_yaw = df_fused["err_fused_yaw_deg"].values
    ax3.plot(t, err_yaw, color="#9467bd", linewidth=1.5, label="Error de Rumbo |ψ_est - ψ_gt| (°)")
    ax3.plot(t, df_fused["sigma_3_yaw_deg"].values, color="#d62728", linestyle="--", linewidth=1.2, label="Cota 3σ Rumbo (°)")
    ax3.set_ylabel("Error Rumbo (°)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Tiempo de simulación (s)", fontsize=11)
    ax3.legend(loc="upper right", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(plots_dir, "sensor_fusion_errors.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  [Gráfico generado] {out_path}")
    return out_path


if __name__ == "__main__":
    imu_f = os.path.join(parent_dir, "data", "imu_measurements.csv")
    vis_f = os.path.join(parent_dir, "data", "visual_navigation_results.csv")
    flight_f = os.path.join(parent_dir, "data", "flight_log.csv")
    meta_f = os.path.join(parent_dir, "data", "reference_map_meta.json")
    run_sensor_fusion(imu_f, vis_f, flight_f, meta_f)
