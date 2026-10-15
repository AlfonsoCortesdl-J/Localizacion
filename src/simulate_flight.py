"""
simulate_flight.py
==================
Motor de simulación de dinámica de vuelo 6-DOF basado en JSBSim para
el desarrollo y evaluación de prototipos de navegación aérea y fusión sensorial.

Marcos de referencia implementados:
----------------------------------
1. Geodésico WGS84:
   - Latitud (grados, -90 a +90, positivo Norte)
   - Longitud (grados, -180 a +180, positivo Este)
   - Altitud MSL (metros sobre el nivel medio del mar)
   - Altitud AGL (metros sobre el terreno)

2. Local Tangente NED (North-East-Down):
   - X_ned: Norte geográfico (m/s)
   - Y_ned: Este geográfico (m/s)
   - Z_ned: Hacia el centro de la Tierra (m/s, descenso positivo)

3. Ejes Cuerpo de la Aeronave (Body Frame FRD: Forward-Right-Down):
   - X_b: A lo largo del eje longitudinal del fuselaje, positivo hacia adelante (nariz).
   - Y_b: A lo largo del ala derecha, positivo hacia estribor.
   - Z_b: Perpendicular al plano X-Y, positivo hacia abajo (piso).

4. Ángulos de Euler (Convención aeroespacial Z-Y-X):
   - Phi (Roll / Alabeo): Rotación sobre X_b [grados o rad]. Positivo: ala derecha abajo.
   - Theta (Pitch / Cabeceo): Rotación sobre Y_b [grados o rad]. Positivo: nariz arriba.
   - Psi (Yaw / Guiñada o Rumbo Verdadero): Rotación sobre Z_b [grados o rad]. Positivo: giro horario visto desde arriba.

Distinción física de aceleraciones:
-----------------------------------
- Fuerza Específica (Specific Force) f_b: Medida real de un acelerómetro inercial triaxial.
  Representa la aceleración debida a todas las fuerzas de no-gravedad (aerodinámicas y propulsión):
  f_b = a_inercial - g
  En vuelo estacionario recto y nivelado, f_z ~ -1g (-9.81 m/s^2), midiendo la sustentación que cancela el peso.
- Aceleración Cinemática (Kinematic / Coordinate Acceleration) dv_b/dt: Derivada temporal de
  las velocidades en ejes cuerpo (udot, vdot, wdot). En vuelo recto nivelado es cero.
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import jsbsim


class FlightSimulator:
    """
    Controlador y puente de simulación JSBSim para integración con sistemas de navegación.
    """

    # Factores de conversión estándar
    FT_TO_M = 0.3048
    M_TO_FT = 1.0 / 0.3048
    KTS_TO_MPS = 0.514444
    MPS_TO_KTS = 1.0 / 0.514444
    FT_SEC2_TO_MPS2 = 0.3048
    GRAVITY_MPS2 = 9.80665

    def __init__(self, scenario_config_path: str):
        """
        Inicializa el simulador a partir de un archivo JSON de configuración.
        """
        if not os.path.exists(scenario_config_path):
            raise FileNotFoundError(f"Archivo de escenario no encontrado: {scenario_config_path}")

        with open(scenario_config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.scenario_name = self.config.get("scenario_name", "escenario_default")
        self.aircraft_model = self.config["aircraft"]["model"]
        self.sim_freq_hz = float(self.config["simulation_settings"]["sim_frequency_hz"])
        self.dt = 1.0 / self.sim_freq_hz

        self.fdm: Optional[jsbsim.FGFDMExec] = None
        self.wall_start_timestamp: float = 0.0
        self.is_initialized = False

    def initialize(self) -> None:
        """
        Configura e inicializa el modelo JSBSim con las condiciones iniciales del escenario.
        """
        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.set_debug_level(0)  # Reducir verbosidad en consola

        # Cargar modelo de aeronave
        model_loaded = self.fdm.load_model(self.aircraft_model)
        if not model_loaded:
            raise RuntimeError(f"No se pudo cargar el modelo de aeronave: {self.aircraft_model}")

        # Configurar paso de integración
        self.fdm.set_dt(self.dt)

        # Establecer condiciones iniciales (IC)
        ic = self.config["initial_conditions"]
        alt_ft = ic["altitude_msl_m"] * self.M_TO_FT

        self.fdm.set_property_value("ic/lat-gc-deg", float(ic["latitude_deg"]))
        self.fdm.set_property_value("ic/long-gc-deg", float(ic["longitude_deg"]))
        self.fdm.set_property_value("ic/h-sl-ft", float(alt_ft))
        self.fdm.set_property_value("ic/vc-kts", float(ic["airspeed_kts"]))
        self.fdm.set_property_value("ic/psi-true-deg", float(ic["heading_deg"]))
        self.fdm.set_property_value("ic/theta-deg", float(ic.get("pitch_deg", 1.5)))
        self.fdm.set_property_value("ic/phi-deg", float(ic.get("roll_deg", 0.0)))

        # Ejecutar inicialización de condiciones de contorno
        self.fdm.run_ic()

        # Configuración del motor y controles iniciales
        if ic.get("engine_running", True):
            self.fdm.set_property_value("propulsion/engine[0]/set-running", 1)
            self.fdm.set_property_value("fcs/mixture-cmd-norm", float(ic.get("mixture", 1.0)))
            self.fdm.set_property_value("fcs/throttle-cmd-norm", float(ic.get("throttle", 0.72)))

        self.wall_start_timestamp = time.time()
        self.is_initialized = True

    def get_target_attitudes(self, sim_time: float) -> tuple[float, float]:
        """
        Retorna las consignas de alabeo (roll) y cabeceo (pitch) según el plan de vuelo.
        """
        maneuvers = self.config.get("flight_plan_maneuvers", [])
        for m in maneuvers:
            if m["start_sec"] <= sim_time < m["end_sec"]:
                return float(m["target_roll_deg"]), float(m["target_pitch_deg"])
        # Fuera de rango, mantener nivelado
        return 0.0, 2.0

    def apply_flight_control(self, sim_time: float) -> None:
        """
        Controlador de estabilización de actitud en bucle cerrado (PD).
        Permite seguir trayectorias reproducibles y suaves sin oscilaciones divergentes.
        """
        roll_target_deg, pitch_target_deg = self.get_target_attitudes(sim_time)

        # Lecturas actuales de actitud y velocidad angular
        phi_deg = self.fdm.get_property_value("attitude/phi-deg")
        theta_deg = self.fdm.get_property_value("attitude/theta-deg")
        p_rad_s = self.fdm.get_property_value("velocities/p-rad_sec")
        q_rad_s = self.fdm.get_property_value("velocities/q-rad_sec")
        r_rad_s = self.fdm.get_property_value("velocities/r-rad_sec")

        # Ley de control de alerones (Control proporcional de ángulo con amortiguamiento giroscópico)
        aileron_cmd = 0.025 * (roll_target_deg - phi_deg) - 0.18 * p_rad_s
        aileron_cmd = max(-0.8, min(0.8, aileron_cmd))
        self.fdm.set_property_value("fcs/aileron-cmd-norm", aileron_cmd)

        # Ley de control de elevador (En aeronaves, elevador negativo produce cabeceo hacia arriba)
        elevator_cmd = -0.035 * (pitch_target_deg - theta_deg) + 0.22 * q_rad_s
        elevator_cmd = max(-0.8, min(0.8, elevator_cmd))
        self.fdm.set_property_value("fcs/elevator-cmd-norm", elevator_cmd)

        # Amortiguador de guiñada / timón de dirección para evitar deslizamiento lateral
        rudder_cmd = 0.25 * r_rad_s
        rudder_cmd = max(-0.5, min(0.5, rudder_cmd))
        self.fdm.set_property_value("fcs/rudder-cmd-norm", rudder_cmd)

    def step(self) -> bool:
        """
        Avanza la simulación un paso temporal dt.
        """
        if not self.is_initialized:
            raise RuntimeError("El simulador no ha sido inicializado. Llama a initialize() primero.")

        sim_time = self.fdm.get_sim_time()
        self.apply_flight_control(sim_time)
        return bool(self.fdm.run())

    def get_ground_truth(self) -> Dict[str, Any]:
        """
        Extrae el vector de estado completo del simulador (Ground Truth)
        organizado con unidades del Sistema Internacional (SI) y convenciones aeroespaciales.
        """
        if not self.is_initialized:
            raise RuntimeError("El simulador no ha sido inicializado.")

        sim_time = self.fdm.get_sim_time()
        current_unix_timestamp = self.wall_start_timestamp + sim_time
        iso_str = datetime.fromtimestamp(current_unix_timestamp, tz=timezone.utc).isoformat()

        # Posición
        lat_geod = self.fdm.get_property_value("position/lat-geod-deg")
        lon_geod = self.fdm.get_property_value("position/long-gc-deg")
        lat_gc = self.fdm.get_property_value("position/lat-gc-deg")
        alt_msl_m = self.fdm.get_property_value("position/h-sl-meters")
        alt_agl_m = self.fdm.get_property_value("position/h-agl-ft") * self.FT_TO_M
        alt_msl_ft = self.fdm.get_property_value("position/h-sl-ft")

        # Velocidades lineales (Marco NED y Marco Cuerpo)
        vel_n_mps = self.fdm.get_property_value("velocities/v-north-fps") * self.FT_TO_M
        vel_e_mps = self.fdm.get_property_value("velocities/v-east-fps") * self.FT_TO_M
        vel_d_mps = self.fdm.get_property_value("velocities/v-down-fps") * self.FT_TO_M
        vel_u_mps = self.fdm.get_property_value("velocities/u-fps") * self.FT_TO_M
        vel_v_mps = self.fdm.get_property_value("velocities/v-fps") * self.FT_TO_M
        vel_w_mps = self.fdm.get_property_value("velocities/w-fps") * self.FT_TO_M

        groundspeed_mps = self.fdm.get_property_value("velocities/vg-fps") * self.FT_TO_M
        airspeed_cas_kts = self.fdm.get_property_value("velocities/vc-kts")
        airspeed_tas_kts = self.fdm.get_property_value("velocities/vtrue-kts")

        # Orientación / Actitud (Ángulos de Euler)
        roll_deg = self.fdm.get_property_value("attitude/phi-deg")
        pitch_deg = self.fdm.get_property_value("attitude/theta-deg")
        yaw_deg = self.fdm.get_property_value("attitude/psi-deg")

        # Velocidades angulares (Medición directa de Giróscopo en Ejes Cuerpo)
        p_rad_s = self.fdm.get_property_value("velocities/p-rad_sec")
        q_rad_s = self.fdm.get_property_value("velocities/q-rad_sec")
        r_rad_s = self.fdm.get_property_value("velocities/r-rad_sec")

        # Aceleraciones: Fuerza Específica (Medición directa de Acelerómetro en Ejes Cuerpo)
        # Nota: a-pilot-x/y/z están en ft/sec^2. Se convierten a m/s^2 y a unidades g.
        accel_x_mps2 = self.fdm.get_property_value("accelerations/a-pilot-x-ft_sec2") * self.FT_SEC2_TO_MPS2
        accel_y_mps2 = self.fdm.get_property_value("accelerations/a-pilot-y-ft_sec2") * self.FT_SEC2_TO_MPS2
        accel_z_mps2 = self.fdm.get_property_value("accelerations/a-pilot-z-ft_sec2") * self.FT_SEC2_TO_MPS2

        accel_x_g = self.fdm.get_property_value("accelerations/n-pilot-x-norm")
        accel_y_g = self.fdm.get_property_value("accelerations/n-pilot-y-norm")
        accel_z_g = self.fdm.get_property_value("accelerations/n-pilot-z-norm")

        # Aceleración Cinemática (Derivada temporal de la velocidad en ejes cuerpo)
        udot_mps2 = self.fdm.get_property_value("accelerations/udot-ft_sec2") * self.FT_SEC2_TO_MPS2
        vdot_mps2 = self.fdm.get_property_value("accelerations/vdot-ft_sec2") * self.FT_SEC2_TO_MPS2
        wdot_mps2 = self.fdm.get_property_value("accelerations/wdot-ft_sec2") * self.FT_SEC2_TO_MPS2

        # Comandos de control
        throttle = self.fdm.get_property_value("fcs/throttle-cmd-norm")
        elevator = self.fdm.get_property_value("fcs/elevator-cmd-norm")
        aileron = self.fdm.get_property_value("fcs/aileron-cmd-norm")
        rudder = self.fdm.get_property_value("fcs/rudder-cmd-norm")

        return {
            # Tiempo
            "sim_time_sec": round(sim_time, 4),
            "timestamp_unix": round(current_unix_timestamp, 4),
            "timestamp_iso": iso_str,
            # Posición Geodésica WGS84
            "lat_geod_deg": lat_geod,
            "lon_geod_deg": lon_geod,
            "lat_gc_deg": lat_gc,
            "altitude_msl_m": alt_msl_m,
            "altitude_agl_m": alt_agl_m,
            "altitude_msl_ft": alt_msl_ft,
            # Velocidades
            "groundspeed_mps": groundspeed_mps,
            "airspeed_cas_kts": airspeed_cas_kts,
            "airspeed_tas_kts": airspeed_tas_kts,
            "vel_north_mps": vel_n_mps,
            "vel_east_mps": vel_e_mps,
            "vel_down_mps": vel_d_mps,
            "vel_u_body_mps": vel_u_mps,
            "vel_v_body_mps": vel_v_mps,
            "vel_w_body_mps": vel_w_mps,
            # Orientación (Euler)
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
            # Giróscopo (Ejes cuerpo)
            "gyro_p_rad_s": p_rad_s,
            "gyro_q_rad_s": q_rad_s,
            "gyro_r_rad_s": r_rad_s,
            # Acelerómetro IMU (Fuerza específica, Ejes cuerpo)
            "accel_specific_x_mps2": accel_x_mps2,
            "accel_specific_y_mps2": accel_y_mps2,
            "accel_specific_z_mps2": accel_z_mps2,
            "accel_specific_x_g": accel_x_g,
            "accel_specific_y_g": accel_y_g,
            "accel_specific_z_g": accel_z_g,
            # Aceleración Cinemática (du/dt, dv/dt, dw/dt)
            "accel_kinematic_udot_mps2": udot_mps2,
            "accel_kinematic_vdot_mps2": vdot_mps2,
            "accel_kinematic_wdot_mps2": wdot_mps2,
            # Mandos de control
            "cmd_throttle": throttle,
            "cmd_elevator": elevator,
            "cmd_aileron": aileron,
            "cmd_rudder": rudder,
        }
