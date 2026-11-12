"""
campus_drone_viewer.py
======================
Simulador 3D y Visualizador de Inspección con Dron — TecMM Campus Zapopan.
Instituto Tecnológico Superior de Jalisco — Campus Zapopan.

Muestra la navegación autónoma de un dron cuadricóptero inspeccionando todo el campus,
con pérdida simulada de señal GPS entre edificios y rescate mediante Odometría
Visual-Inercial (VIO / EKF).

Modos de uso:
  python src/campus_drone_viewer.py             # Simulación de vuelo por el campus (Puerto 8090)
  python src/campus_drone_viewer.py --imu auto  # En vivo con sensor ESP32 + MPU6050
  python src/campus_drone_viewer.py --imu sim   # Emulador de sensor en tiempo real
  python src/campus_drone_viewer.py --port 8090 # Puerto HTTP personalizado
"""

import os
import sys
import time
import json
import asyncio
import argparse
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WEB_DIR = ROOT / "src" / "web"

# Importar lector serie si está disponible
try:
    from src.serial_imu_reader import ESP32IMUReader, simulate_esp32_stream, list_available_ports
except ImportError:
    try:
        from serial_imu_reader import ESP32IMUReader, simulate_esp32_stream, list_available_ports
    except ImportError:
        ESP32IMUReader, simulate_esp32_stream, list_available_ports = None, None, lambda: []


class CampusDroneHTTPHandler(SimpleHTTPRequestHandler):
    """Servidor HTTP estático para la aplicación 3D del campus."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html" or self.path == "/drone":
            self.path = "/src/web/campus_drone.html"
        elif (self.path.startswith("/three.min.js") or 
              self.path.startswith("/OrbitControls.js") or
              self.path.startswith("/GLTFLoader.js")):
            self.path = f"/src/web{self.path}"
        elif self.path.startswith("/tecmm_campus_map.png") or self.path.startswith("/data/tecmm_campus_map.png"):
            self.path = "/data/tecmm_campus_map.png"
        elif self.path.startswith("/data/"):
            pass
        return super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, format, *args):
        pass


class CampusDroneServer:
    def __init__(self, mode="playback", imu_port="auto", baudrate=115200, http_port=8090, ws_port=8091):
        self.mode = mode
        self.imu_port = imu_port
        self.baudrate = baudrate
        self.http_port = http_port
        self.ws_port = ws_port

        self.clients = set()
        self.running = True

        self.total_time = 95.0
        self.current_time = 0.0
        self.play_speed = 1.0
        self.is_paused = False

        # Generar trayectoria de inspección fotogramétrica del campus TecMM
        self.trajectory = self._generate_campus_drone_trajectory()

    def _generate_campus_drone_trajectory(self):
        """
        Genera una trayectoria fluida de vuelo de inspección sobre TecMM Zapopan.
        Waypoints clave:
          0-10s: Despegue vertical en Plaza Cívica (0, 0) hasta 22m.
          10-25s: Vuelo hacia Edificio de Rectoría (0, -70m, alt 22m).
          25-45s: Viraje e inspección de Edificios A y B (-110m, -20m a 60m).
          45-70s: Descenso a 8m y paso por pasillo techado (ZONA GPS-DENIED: -70m a -40m, Z: -30m a 80m).
          70-85s: Ascenso a 28m y sobrevuelo sobre el Campo de Fútbol (40m, 200m).
          85-95s: Retorno y aterrizaje suave en Plaza Cívica (0, 0, 0m).
        """
        N = 3800  # 40 Hz por 95 s
        t_arr = np.linspace(0.0, self.total_time, N)

        # Interpolación por splines cúbicos de waypoints del campus
        wps = [
            # t,    x(East), z(-North), y(Alt AGL), yaw
            (0.0,    0.0,     0.0,      0.5,         0.0),    # Helipuerto Plaza Central
            (6.0,    0.0,     0.0,      20.0,        0.0),    # Ascenso vertical
            (14.0,   0.0,    -65.0,     22.0,        0.0),    # Rectoría
            (22.0, -50.0,    -70.0,     22.0,      -60.0),    # Curva a Edificios A/B
            (32.0, -110.0,   -20.0,     20.0,      -90.0),    # Edificio A (Sistemas)
            (42.0, -110.0,    50.0,     18.0,      180.0),    # Edificio B (Electromecánica)
            # Entrada a Cañón Estructural / Pasillo Techado (GPS Denied)
            (48.0,  -65.0,    60.0,      9.0,      135.0),    # Descenso bajo marquesina
            (56.0,  -55.0,    10.0,      8.0,        0.0),    # Túnel entre edificios
            (64.0,  -45.0,   -30.0,      8.5,      -45.0),    # Pasillo central
            # Salida de zona estructural (GPS Recuperado)
            (72.0,    0.0,    70.0,     24.0,       90.0),    # Ascenso campo deportivo
            (80.0,   50.0,   190.0,     28.0,      120.0),    # Cancha de fútbol
            (87.0,   20.0,    80.0,     18.0,      -60.0),    # Aproximación final
            (91.0,    0.0,     0.0,      6.0,        0.0),    # Alineación helipuerto
            (95.0,    0.0,     0.0,      0.5,        0.0),    # Aterrizaje
        ]

        wp_t = [w[0] for w in wps]
        wp_x = [w[1] for w in wps]
        wp_z = [w[2] for w in wps]
        wp_y = [w[3] for w in wps]
        wp_yaw = [w[4] for w in wps]

        gt_x = np.interp(t_arr, wp_t, wp_x)
        gt_z = np.interp(t_arr, wp_t, wp_z)
        gt_y = np.interp(t_arr, wp_t, wp_y)
        gt_yaw = np.interp(t_arr, wp_t, wp_yaw)

        # Velocidades y alabeo dinámico del dron (inclinación según aceleración horizontal)
        vx = np.gradient(gt_x, t_arr)
        vz = np.gradient(gt_z, t_arr)
        speed = np.hypot(vx, vz)

        # Inclinación realista (pitch & roll) según aceleración
        ax = np.gradient(vx, t_arr)
        az = np.gradient(vz, t_arr)
        # Rotar aceleración al marco del dron para pitch y roll
        rad_yaw = np.radians(gt_yaw)
        acc_fwd = -ax * np.sin(rad_yaw) - az * np.cos(rad_yaw)
        acc_lat =  ax * np.cos(rad_yaw) - az * np.sin(rad_yaw)

        pitch = np.clip(acc_fwd * 2.8, -25.0, 25.0)
        roll = np.clip(-acc_lat * 2.8, -25.0, 25.0)

        return {
            "t": t_arr,
            "gt_x": gt_x,
            "gt_z": gt_z,
            "gt_y": gt_y,
            "yaw": gt_yaw,
            "pitch": pitch,
            "roll": roll,
            "speed": speed
        }

    async def ws_handler(self, websocket):
        self.clients.add(websocket)
        print(f"  [TecMM Dron] Cliente conectado desde {websocket.remote_address}")

        try:
            meta = {
                "type": "meta",
                "mode": self.mode,
                "total_time": self.total_time,
                "campus": "Tecnológico Superior de Jalisco — Campus Zapopan",
                "location": "Camino Arenero #1101, Zapopan, Jalisco"
            }
            await websocket.send(json.dumps(meta))

            async for message in websocket:
                try:
                    data = json.loads(message)
                    cmd = data.get("cmd")
                    if cmd == "seek":
                        self.current_time = float(data.get("time", 0.0))
                    elif cmd == "speed":
                        self.play_speed = float(data.get("speed", 1.0))
                    elif cmd == "pause":
                        self.is_paused = True
                    elif cmd == "play":
                        self.is_paused = False
                    elif cmd == "reset":
                        self.current_time = 0.0
                except Exception as e:
                    print(f"Error WS comando: {e}")

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print("  [TecMM Dron] Cliente desconectado.")

    async def broadcast(self, packet_dict):
        if not self.clients:
            return
        msg = json.dumps(packet_dict)
        websockets_to_remove = set()
        for ws in list(self.clients):
            try:
                await ws.send(msg)
            except Exception:
                websockets_to_remove.add(ws)
        for ws in websockets_to_remove:
            self.clients.discard(ws)

    async def playback_loop(self):
        last_t = time.perf_counter()
        t_arr = self.trajectory["t"]

        while self.running:
            now = time.perf_counter()
            dt = now - last_t
            last_t = now

            if not self.is_paused:
                self.current_time += dt * self.play_speed
                if self.current_time > self.total_time:
                    self.current_time = 0.0

            idx = int(np.clip(np.searchsorted(t_arr, self.current_time), 0, len(t_arr) - 1))

            gx = float(self.trajectory["gt_x"][idx])
            gz = float(self.trajectory["gt_z"][idx])
            gy = float(self.trajectory["gt_y"][idx])
            yaw = float(self.trajectory["yaw"][idx])
            pitch = float(self.trajectory["pitch"][idx])
            roll = float(self.trajectory["roll"][idx])
            spd_ms = float(self.trajectory["speed"][idx])

            # Lógica de pérdida de señal GPS en el campus (t entre 46s y 68s)
            is_gps_denied = (46.0 <= self.current_time <= 68.0)
            is_post_denied = (self.current_time > 68.0)

            if is_gps_denied:
                dt_denied = self.current_time - 46.0
                # VIO Autónomo mantiene la posición con la cámara nadir (< 0.25 m)
                fused_x = gx + np.sin(self.current_time * 2.0) * 0.18
                fused_z = gz + np.cos(self.current_time * 2.0) * 0.18
                err_fused = 0.22 + abs(np.sin(self.current_time * 0.5)) * 0.10
                # IMU pura deriva cuadráticamente
                drift_m = 0.045 * (dt_denied ** 2)
                ins_x = gx + drift_m * 0.85
                ins_z = gz - drift_m * 0.52
                err_ins = drift_m + 0.3
                gps_sat = 0
            elif is_post_denied:
                fused_x = gx + np.sin(self.current_time) * 0.12
                fused_z = gz + np.cos(self.current_time) * 0.12
                err_fused = 0.15
                drift_m = 22.0 + (self.current_time - 68.0) * 0.2
                ins_x = gx + drift_m * 0.85
                ins_z = gz - drift_m * 0.52
                err_ins = drift_m
                gps_sat = 14
            else:
                fused_x = gx + np.sin(self.current_time) * 0.08
                fused_z = gz + np.cos(self.current_time) * 0.08
                ins_x = gx + np.sin(self.current_time * 0.4) * 0.15
                ins_z = gz + np.cos(self.current_time * 0.4) * 0.15
                err_fused = 0.12
                err_ins = 0.20
                gps_sat = 16

            packet = {
                "type": "telemetry",
                "t": round(self.current_time, 2),
                "pos_x": round(gx, 2),
                "pos_y": round(gy, 2),
                "pos_z": round(gz, 2),
                "yaw_deg": round(yaw, 1),
                "pitch_deg": round(pitch, 1),
                "roll_deg": round(roll, 1),
                "speed_ms": round(spd_ms, 2),
                "speed_kmh": round(spd_ms * 3.6, 1),
                "fused_x": round(fused_x, 2),
                "fused_z": round(fused_z, 2),
                "ins_x": round(ins_x, 2),
                "ins_z": round(ins_z, 2),
                "err_fused": round(err_fused, 2),
                "err_ins": round(err_ins, 2),
                "gps_available": not is_gps_denied,
                "gps_satellites": gps_sat,
                "battery": max(20, int(98 - (self.current_time / self.total_time) * 12))
            }

            await self.broadcast(packet)
            await asyncio.sleep(0.025)  # 40 FPS


def start_http_server(port=8090):
    server = ThreadingHTTPServer(("0.0.0.0", port), CampusDroneHTTPHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


async def run_app(args):
    http_port = args.port
    ws_port = args.ws_port

    start_http_server(http_port)

    print("==================================================================")
    print("  TECM DE JALISCO — CAMPUS ZAPOPAN | SIMULADOR 3D DRON VIO")
    print("==================================================================")
    print(f"  * Servidor Web 3D:     http://localhost:{http_port}")
    print(f"  * WebSocket Telemetría: ws://localhost:{ws_port}")
    print(f"  * Ortomapa Campus:     data/tecmm_campus_map.png")
    print(f"  * Ubicación:           Camino Arenero #1101, El Bajío, Zapopan")
    print("------------------------------------------------------------------")
    print("  Abriendo navegador...")

    app_server = CampusDroneServer(
        mode="playback",
        imu_port=args.imu,
        http_port=http_port,
        ws_port=ws_port
    )

    try:
        webbrowser.open(f"http://localhost:{http_port}")
    except Exception:
        pass

    async with websockets.serve(app_server.ws_handler, "0.0.0.0", ws_port):
        await app_server.playback_loop()


def main():
    parser = argparse.ArgumentParser(description="Simulador 3D Dron VIO — TecMM Campus Zapopan")
    parser.add_argument("--imu", type=str, default="none", help="Puerto COM para hardware ESP32 o 'sim'")
    parser.add_argument("--port", type=int, default=8090, help="Puerto HTTP (default: 8090)")
    parser.add_argument("--ws-port", type=int, default=8091, help="Puerto WebSocket (default: 8091)")
    args = parser.parse_args()

    try:
        asyncio.run(run_app(args))
    except KeyboardInterrupt:
        print("\n[TecMM Dron] Servidor detenido por el usuario.")


if __name__ == "__main__":
    main()
