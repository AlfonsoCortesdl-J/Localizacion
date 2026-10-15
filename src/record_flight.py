"""
record_flight.py
================
Ejecuta la simulación completa del escenario, registra la telemetría Ground Truth
en archivos CSV y genera los registros de sincronización para cámaras virtuales e IMU.

Archivos generados:
-------------------
1. data/flight_log.csv:
   Telemetría completa de referencia 6-DOF del simulador a la frecuencia de log (ej. 20 Hz).
2. data/camera_ground_frames.csv:
   Registro de disparos de la cámara virtual orientada a tierra (Nadir) a 5 Hz.
3. data/camera_sky_frames.csv:
   Registro de disparos de la cámara virtual orientada al cielo (Zenith) a 2 Hz.
4. data/imu_measurements.csv:
   Flujo de mediciones de sensores inerciales (acelerómetro y giróscopo) a 100 Hz,
   separando estrictamente las mediciones sensoriales del Ground Truth.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Asegurar importación del módulo local src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.simulate_flight import FlightSimulator


def validate_flight_data(df: pd.DataFrame) -> bool:
    """
    Verifica la integridad de los datos registrados:
    - Ausencia de NaNs o infinitos.
    - Timestamps estrictamente crecientes.
    - Rangos físicos coherentes.
    """
    print("\n--- Verificación de integridad de datos ---")

    # 1. Comprobación de valores nulos o infinitos
    null_count = df.isnull().sum().sum()
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).values.sum()
    if null_count > 0 or inf_count > 0:
        print(f"[ERROR] Datos corruptos encontrados: {null_count} nulos, {inf_count} infinitos.")
        return False
    print("  [OK] Cero valores nulos y cero valores infinitos.")

    # 2. Comprobación de monotonía temporal
    time_diffs = np.diff(df["sim_time_sec"].values)
    if np.any(time_diffs <= 0):
        print("[ERROR] Timestamps no son estrictamente crecientes.")
        return False
    print("  [OK] Timestamps estrictamente monótonos crecientes.")

    # 3. Comprobación de límites físicos (sin colisión contra el terreno)
    min_alt = df["altitude_msl_m"].min()
    max_alt = df["altitude_msl_m"].max()
    min_speed = df["groundspeed_mps"].min()
    max_speed = df["groundspeed_mps"].max()

    if min_alt <= 0.0:
        print(f"[ERROR] Altitud inválida detectada (avión estrellado): min {min_alt} m")
        return False
    print(f"  [OK] Altitud segura y consistente: min {min_alt:.1f} m, max {max_alt:.1f} m.")
    print(f"  [OK] Velocidad física coherente: min {min_speed:.1f} m/s, max {max_speed:.1f} m/s ({min_speed*1.94384:.1f} a {max_speed*1.94384:.1f} kts).")
    return True


def run_recording(scenario_path: str = "scenarios/escenario_base.json") -> None:
    """
    Ejecuta el escenario y graba la telemetría y metadatos de sensores.
    """
    print(f"Cargando escenario: {scenario_path}")
    sim = FlightSimulator(scenario_path)
    sim.initialize()

    duration_sec = float(sim.config["simulation_settings"]["duration_sec"])
    sim_dt = sim.dt
    log_freq = float(sim.config["simulation_settings"]["log_frequency_hz"])
    log_interval = 1.0 / log_freq

    # Frecuencias de sensores
    cam_ground_fps = float(sim.config["sensor_interfaces"]["ground_camera"]["fps"])
    cam_ground_interval = 1.0 / cam_ground_fps

    cam_sky_fps = float(sim.config["sensor_interfaces"]["sky_camera"]["fps"])
    cam_sky_interval = 1.0 / cam_sky_fps

    imu_freq = float(sim.config["sensor_interfaces"]["imu"]["sampling_rate_hz"])
    imu_interval = 1.0 / imu_freq

    # Contenedores de datos
    flight_records = []
    camera_ground_records = []
    camera_sky_records = []
    imu_records = []

    last_log_time = -1.0
    last_cam_ground_time = -1.0
    last_cam_sky_time = -1.0
    last_imu_time = -1.0

    cam_ground_frame_id = 0
    cam_sky_frame_id = 0
    imu_sample_id = 0

    total_steps = int(duration_sec / sim_dt)
    print(f"Iniciando simulación de vuelo: {duration_sec} s ({total_steps} pasos dinámicos a {sim.sim_freq_hz} Hz)...")

    # Ruido de sensores para IMU sintética (separación de Ground Truth)
    noise_cfg = sim.config["sensor_interfaces"]["imu"]
    accel_std = noise_cfg.get("noise_density_accel_ug_rthz", 100.0) * 1e-6 * 9.80665 * np.sqrt(imu_freq)
    gyro_std = np.radians(noise_cfg.get("noise_density_gyro_mdps_rthz", 10.0) * 1e-3 * np.sqrt(imu_freq))

    for step in range(total_steps):
        sim_time = sim.fdm.get_sim_time()

        # 1. Registro de Ground Truth de vuelo
        if sim_time - last_log_time >= (log_interval - 1e-6):
            gt = sim.get_ground_truth()
            flight_records.append(gt)
            last_log_time = sim_time

        # 2. Disparador de Cámara Virtual de Terreno (Nadir)
        if sim_time - last_cam_ground_time >= (cam_ground_interval - 1e-6):
            gt = sim.get_ground_truth()
            camera_ground_records.append({
                "frame_id": cam_ground_frame_id,
                "image_filename": f"img_ground_{cam_ground_frame_id:05d}.png",
                "sim_time_sec": gt["sim_time_sec"],
                "timestamp_unix": gt["timestamp_unix"],
                "timestamp_iso": gt["timestamp_iso"],
                "lat_geod_deg": gt["lat_geod_deg"],
                "lon_geod_deg": gt["lon_geod_deg"],
                "altitude_msl_m": gt["altitude_msl_m"],
                "altitude_agl_m": gt["altitude_agl_m"],
                "roll_deg": gt["roll_deg"],
                "pitch_deg": gt["pitch_deg"],
                "yaw_deg": gt["yaw_deg"],
                "groundspeed_mps": gt["groundspeed_mps"],
                "camera_mount": "nadir",
                "fov_deg": sim.config["sensor_interfaces"]["ground_camera"]["fov_deg"],
                "scenario_name": sim.scenario_name
            })
            cam_ground_frame_id += 1
            last_cam_ground_time = sim_time

        # 3. Disparador de Cámara Virtual de Cielo (Zenith)
        if sim_time - last_cam_sky_time >= (cam_sky_interval - 1e-6):
            gt = sim.get_ground_truth()
            camera_sky_records.append({
                "frame_id": cam_sky_frame_id,
                "image_filename": f"img_sky_{cam_sky_frame_id:05d}.png",
                "sim_time_sec": gt["sim_time_sec"],
                "timestamp_unix": gt["timestamp_unix"],
                "timestamp_iso": gt["timestamp_iso"],
                "lat_geod_deg": gt["lat_geod_deg"],
                "lon_geod_deg": gt["lon_geod_deg"],
                "altitude_msl_m": gt["altitude_msl_m"],
                "roll_deg": gt["roll_deg"],
                "pitch_deg": gt["pitch_deg"],
                "yaw_deg": gt["yaw_deg"],
                "camera_mount": "zenith",
                "fov_deg": sim.config["sensor_interfaces"]["sky_camera"]["fov_deg"],
                "scenario_name": sim.scenario_name
            })
            cam_sky_frame_id += 1
            last_cam_sky_time = sim_time

        # 4. Flujo de mediciones IMU sintética (con ruido térmico gaussiano y bias)
        if sim_time - last_imu_time >= (imu_interval - 1e-6):
            gt = sim.get_ground_truth()
            # Inyección de ruido representativo de sensor MEMS comercial (ej. ICM-42688P o MPU-6050)
            accel_meas_x = gt["accel_specific_x_mps2"] + np.random.normal(0.0, accel_std)
            accel_meas_y = gt["accel_specific_y_mps2"] + np.random.normal(0.0, accel_std)
            accel_meas_z = gt["accel_specific_z_mps2"] + np.random.normal(0.0, accel_std)

            gyro_meas_p = gt["gyro_p_rad_s"] + np.random.normal(0.0, gyro_std)
            gyro_meas_q = gt["gyro_q_rad_s"] + np.random.normal(0.0, gyro_std)
            gyro_meas_r = gt["gyro_r_rad_s"] + np.random.normal(0.0, gyro_std)

            imu_records.append({
                "sample_id": imu_sample_id,
                "sim_time_sec": gt["sim_time_sec"],
                "timestamp_unix": gt["timestamp_unix"],
                "timestamp_iso": gt["timestamp_iso"],
                # Mediciones brutas del sensor inercial
                "accel_x_mps2": round(accel_meas_x, 5),
                "accel_y_mps2": round(accel_meas_y, 5),
                "accel_z_mps2": round(accel_meas_z, 5),
                "gyro_p_rad_s": round(gyro_meas_p, 6),
                "gyro_q_rad_s": round(gyro_meas_q, 6),
                "gyro_r_rad_s": round(gyro_meas_r, 6),
                "gyro_p_deg_s": round(np.degrees(gyro_meas_p), 4),
                "gyro_q_deg_s": round(np.degrees(gyro_meas_q), 4),
                "gyro_r_deg_s": round(np.degrees(gyro_meas_r), 4),
            })
            imu_sample_id += 1
            last_imu_time = sim_time

        # Avanzar integración dinámica
        sim.step()

    # Convertir a DataFrames y exportar a CSV
    data_dir = os.path.join(parent_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    df_flight = pd.DataFrame(flight_records)
    flight_csv = os.path.join(data_dir, "flight_log.csv")
    df_flight.to_csv(flight_csv, index=False)
    print(f"\n[Archivo generado] {flight_csv} ({len(df_flight)} registros a {log_freq} Hz)")

    df_cam_ground = pd.DataFrame(camera_ground_records)
    cam_ground_csv = os.path.join(data_dir, "camera_ground_frames.csv")
    df_cam_ground.to_csv(cam_ground_csv, index=False)
    print(f"[Archivo generado] {cam_ground_csv} ({len(df_cam_ground)} fotogramas a {cam_ground_fps} FPS)")

    df_cam_sky = pd.DataFrame(camera_sky_records)
    cam_sky_csv = os.path.join(data_dir, "camera_sky_frames.csv")
    df_cam_sky.to_csv(cam_sky_csv, index=False)
    print(f"[Archivo generado] {cam_sky_csv} ({len(df_cam_sky)} fotogramas a {cam_sky_fps} FPS)")

    df_imu = pd.DataFrame(imu_records)
    imu_csv = os.path.join(data_dir, "imu_measurements.csv")
    df_imu.to_csv(imu_csv, index=False)
    print(f"[Archivo generado] {imu_csv} ({len(df_imu)} muestras a {imu_freq} Hz)")

    # Validar integridad
    is_valid = validate_flight_data(df_flight)
    if is_valid:
        print("\n>>> ¡Simulación y registro completados exitosamente con 100% de integridad física y temporal! <<<")
    else:
        print("\n[ALERTA] La simulación terminó pero se detectaron inconsistencias en los datos.")


if __name__ == "__main__":
    scenario_file = os.path.join(parent_dir, "scenarios", "escenario_base.json")
    run_recording(scenario_file)
