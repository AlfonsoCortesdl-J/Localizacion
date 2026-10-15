"""
plot_results.py
===============
Genera gráficos de ingeniería de alta resolución a partir de los datos registrados
en data/flight_log.csv para el análisis de dinámica de vuelo, sistemas de navegación
y evaluación de algoritmos de estimación de estado.

Gráficos generados en la carpeta plots/:
----------------------------------------
1. trajectory.png:
   - Vista 2D en plano horizontal (Longitud vs Latitud) con vectores de rumbo y fases de vuelo.
   - Vista 3D (Latitud, Longitud, Altitud MSL).
2. altitude.png:
   - Perfil temporal de altitud MSL (metros y pies).
   - Velocidad vertical (tasa de ascenso/descenso, V_down).
3. velocity.png:
   - Comparación de velocidad verdadera (TAS), indicada (CAS) y respecto al suelo (Groundspeed).
   - Componentes del vector velocidad en marco local NED (North, East, Down).
4. attitude.png:
   - Ángulos de Euler de orientación (Roll, Pitch, Yaw).
   - Velocidades angulares en ejes cuerpo medidas por giróscopo (p, q, r).
5. imu_accelerations.png:
   - Fuerza específica medida por acelerómetro (a_x, a_y, a_z) vs derivadas cinemáticas (udot, vdot, wdot).
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Estilo visual limpio y profesional para ingeniería
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_trajectory(df: pd.DataFrame, output_dir: str) -> str:
    """Genera gráfico 2D y 3D de la trayectoria de vuelo."""
    fig = plt.figure(figsize=(14, 6))

    # Subplot 1: Trayectoria 2D (Plano Horizontal)
    ax1 = fig.add_subplot(1, 2, 1)
    lons = df["lon_geod_deg"].values
    lats = df["lat_geod_deg"].values
    times = df["sim_time_sec"].values

    scatter = ax1.scatter(lons, lats, c=times, cmap="viridis", s=15, alpha=0.9, zorder=3)
    ax1.plot(lons, lats, color="#444444", linewidth=1.2, linestyle="--", alpha=0.6, zorder=2)

    # Marcador de inicio y fin
    ax1.plot(lons[0], lats[0], marker="o", color="#2ca02c", markersize=9, label="Punto Inicial (t=0s)", zorder=4)
    ax1.plot(lons[-1], lats[-1], marker="s", color="#d62728", markersize=9, label=f"Punto Final (t={times[-1]:.0f}s)", zorder=4)

    # Flechas de dirección a lo largo de la ruta
    arrow_indices = np.linspace(10, len(df) - 10, 6, dtype=int)
    for idx in arrow_indices:
        dlon = lons[idx + 1] - lons[idx - 1]
        dlat = lats[idx + 1] - lats[idx - 1]
        ax1.annotate("", xy=(lons[idx] + dlon * 2, lats[idx] + dlat * 2), xytext=(lons[idx], lats[idx]),
                     arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.8), zorder=5)

    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label("Tiempo de simulación (s)")
    ax1.set_title("Trayectoria en Tierra (Ground Track 2D)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Longitud WGS84 (°)", fontsize=10)
    ax1.set_ylabel("Latitud WGS84 (°)", fontsize=10)
    ax1.legend(loc="best", frameon=True)
    ax1.ticklabel_format(useOffset=False)

    # Subplot 2: Trayectoria 3D
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    alts = df["altitude_msl_m"].values
    p3d = ax2.scatter(lons, lats, alts, c=times, cmap="viridis", s=12, alpha=0.8)
    ax2.plot(lons, lats, alts, color="#333333", linewidth=1.0, linestyle=":")

    ax2.scatter(lons[0], lats[0], alts[0], color="#2ca02c", s=60, label="Inicio")
    ax2.scatter(lons[-1], lats[-1], alts[-1], color="#d62728", s=60, label="Fin")

    ax2.set_title("Trayectoria Tridimensional (3D Flight Path)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Longitud (°)", labelpad=8)
    ax2.set_ylabel("Latitud (°)", labelpad=8)
    ax2.set_zlabel("Altitud MSL (m)", labelpad=8)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "trajectory.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_altitude(df: pd.DataFrame, output_dir: str) -> str:
    """Genera gráfico del perfil de altitud y velocidad vertical."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    t = df["sim_time_sec"].values

    # Altitud en metros y pies
    ax1.plot(t, df["altitude_msl_m"].values, color="#1f77b4", linewidth=2.0, label="Altitud MSL (metros)")
    ax1.set_ylabel("Altitud (m)", color="#1f77b4", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_title("Perfil Temporal de Altitud y Tasa de Ascenso/Descenso", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Eje secundario para pies
    ax1_ft = ax1.twinx()
    ax1_ft.plot(t, df["altitude_msl_ft"].values, color="#ff7f0e", linewidth=1.0, linestyle="--", alpha=0.7, label="Altitud MSL (pies)")
    ax1_ft.set_ylabel("Altitud (ft)", color="#ff7f0e", fontsize=11)
    ax1_ft.tick_params(axis="y", labelcolor="#ff7f0e")

    # Velocidad vertical V_down (NED) -> Tasa de ascenso = -V_down
    climb_rate_mps = -df["vel_down_mps"].values
    ax2.plot(t, climb_rate_mps, color="#2ca02c", linewidth=1.8, label="Tasa de ascenso (-V_down)")
    ax2.axhline(0.0, color="#888888", linestyle=":", linewidth=1.2)
    ax2.set_ylabel("Tasa vertical (m/s)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Tiempo de simulación (s)", fontsize=11)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "altitude.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_velocity(df: pd.DataFrame, output_dir: str) -> str:
    """Genera comparación de velocidades aerodinámicas y componentes NED."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    t = df["sim_time_sec"].values

    # Velocidades escalares
    ax1.plot(t, df["groundspeed_mps"].values, label="Groundspeed V_g (m/s)", color="#1f77b4", linewidth=2.0)
    ax1.plot(t, df["airspeed_tas_kts"].values * 0.514444, label="True Airspeed TAS (m/s)", color="#ff7f0e", linewidth=1.8, linestyle="--")
    ax1.plot(t, df["airspeed_cas_kts"].values * 0.514444, label="Calibrated Airspeed CAS (m/s)", color="#2ca02c", linewidth=1.5, linestyle=":")
    ax1.set_ylabel("Velocidad (m/s)", fontsize=11, fontweight="bold")
    ax1.set_title("Evolución de Velocidades de Vuelo y Componentes NED", fontsize=13, fontweight="bold")
    ax1.legend(loc="best", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Componentes NED
    ax2.plot(t, df["vel_north_mps"].values, label="V_North (m/s)", color="#d62728", linewidth=1.8)
    ax2.plot(t, df["vel_east_mps"].values, label="V_East (m/s)", color="#9467bd", linewidth=1.8)
    ax2.plot(t, df["vel_down_mps"].values, label="V_Down (m/s)", color="#8c564b", linewidth=1.8)
    ax2.set_ylabel("Velocidad NED (m/s)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Tiempo de simulación (s)", fontsize=11)
    ax2.legend(loc="best", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "velocity.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_attitude(df: pd.DataFrame, output_dir: str) -> str:
    """Genera gráfico de ángulos de Euler y velocidades angulares (giróscopo)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    t = df["sim_time_sec"].values

    # Ángulos de Euler
    ax1.plot(t, df["roll_deg"].values, label="Roll / Alabeo Φ (°)", color="#1f77b4", linewidth=2.0)
    ax1.plot(t, df["pitch_deg"].values, label="Pitch / Cabeceo θ (°)", color="#ff7f0e", linewidth=2.0)
    ax1.plot(t, df["yaw_deg"].values, label="Yaw / Rumbo ψ (°)", color="#2ca02c", linewidth=2.0)
    ax1.set_ylabel("Actitud Euler (°)", fontsize=11, fontweight="bold")
    ax1.set_title("Orientación de la Aeronave y Mediciones de Velocidad Angular", fontsize=13, fontweight="bold")
    ax1.legend(loc="best", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Velocidades angulares (Giróscopo en ejes cuerpo)
    p_deg_s = np.degrees(df["gyro_p_rad_s"].values)
    q_deg_s = np.degrees(df["gyro_q_rad_s"].values)
    r_deg_s = np.degrees(df["gyro_r_rad_s"].values)

    ax2.plot(t, p_deg_s, label="p (Roll rate, °/s)", color="#1f77b4", linewidth=1.5, alpha=0.9)
    ax2.plot(t, q_deg_s, label="q (Pitch rate, °/s)", color="#ff7f0e", linewidth=1.5, alpha=0.9)
    ax2.plot(t, r_deg_s, label="r (Yaw rate, °/s)", color="#2ca02c", linewidth=1.5, alpha=0.9)
    ax2.axhline(0.0, color="#888888", linestyle=":", linewidth=1.0)
    ax2.set_ylabel("Giróscopo (°/s)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Tiempo de simulación (s)", fontsize=11)
    ax2.legend(loc="best", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "attitude.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_imu_accelerations(df: pd.DataFrame, output_dir: str) -> str:
    """
    Genera comparación detallada entre la Fuerza Específica (medición real de acelerómetro)
    y la Aceleración Cinemática pura (derivada temporal de la velocidad en ejes cuerpo).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    t = df["sim_time_sec"].values

    # 1. Fuerza Específica (Acelerómetro IMU)
    ax1.plot(t, df["accel_specific_x_mps2"].values, label="f_x (Acelerómetro X cuerpo)", color="#1f77b4", linewidth=1.8)
    ax1.plot(t, df["accel_specific_y_mps2"].values, label="f_y (Acelerómetro Y cuerpo)", color="#2ca02c", linewidth=1.8)
    ax1.plot(t, df["accel_specific_z_mps2"].values, label="f_z (Acelerómetro Z cuerpo, ~ -1g)", color="#d62728", linewidth=1.8)
    ax1.set_ylabel("Fuerza Específica (m/s²)", fontsize=11, fontweight="bold")
    ax1.set_title("Fuerza Específica IMU (f = a - g) vs Derivadas Cinemáticas (du/dt)", fontsize=13, fontweight="bold")
    ax1.legend(loc="best", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Aceleración Cinemática en ejes cuerpo (udot, vdot, wdot)
    ax2.plot(t, df["accel_kinematic_udot_mps2"].values, label="udot = du/dt (m/s²)", color="#1f77b4", linestyle="--", linewidth=1.5)
    ax2.plot(t, df["accel_kinematic_vdot_mps2"].values, label="vdot = dv/dt (m/s²)", color="#2ca02c", linestyle="--", linewidth=1.5)
    ax2.plot(t, df["accel_kinematic_wdot_mps2"].values, label="wdot = dw/dt (m/s²)", color="#d62728", linestyle="--", linewidth=1.5)
    ax2.axhline(0.0, color="#888888", linestyle=":", linewidth=1.0)
    ax2.set_ylabel("Acel. Cinemática (m/s²)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Tiempo de simulación (s)", fontsize=11)
    ax2.legend(loc="best", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "imu_accelerations.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def generate_all_plots(log_csv_path: str = "data/flight_log.csv", plots_dir: str = "plots") -> None:
    """Genera el paquete completo de gráficas de la simulación."""
    if not os.path.exists(log_csv_path):
        raise FileNotFoundError(f"Archivo de log no encontrado: {log_csv_path}. Ejecuta primero record_flight.py.")

    os.makedirs(plots_dir, exist_ok=True)
    print(f"Cargando datos para graficación desde: {log_csv_path}...")
    df = pd.read_csv(log_csv_path)

    p1 = plot_trajectory(df, plots_dir)
    print(f"  [Gráfico generado] {p1}")

    p2 = plot_altitude(df, plots_dir)
    print(f"  [Gráfico generado] {p2}")

    p3 = plot_velocity(df, plots_dir)
    print(f"  [Gráfico generado] {p3}")

    p4 = plot_attitude(df, plots_dir)
    print(f"  [Gráfico generado] {p4}")

    p5 = plot_imu_accelerations(df, plots_dir)
    print(f"  [Gráfico generado] {p5}")

    print("\n>>> ¡Todas las gráficas de análisis fueron generadas exitosamente en la carpeta plots/! <<<")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    csv_file = os.path.join(parent_dir, "data", "flight_log.csv")
    plots_path = os.path.join(parent_dir, "plots")
    generate_all_plots(csv_file, plots_path)
