"""
serial_imu_reader.py
====================
Controlador y receptor serie en tiempo real para IMU fisica:
- Soporte para ESP32 + MPU6050 (por USB UART a 115200 baudios o Bluetooth SPP "AvionESP32").
- Soporte para STM32 / Arduino con tramas JSON y Binarias.

Protocolo JSON recibido desde ESP32:
  {"t":12345,"ax":0.05,"ay":-0.02,"az":9.81,"gx":0.01,"gy":-0.01,"gz":0.02}

Caracteristicas:
  1. Autodeteccion y escaneo de puertos COM en Windows (USB y Bluetooth Serial).
  2. Filtro complementario de actitud 6-DOF (Roll, Pitch, Yaw) integrado en tiempo real a 50 Hz.
  3. Descarte automatico de mensajes de depuracion del microcontrolador (lineas con '#').
  4. Modo emulador (--simulate) para probar todo el flujo sin conectar hardware.
  5. Registro automatico en data/physical_imu_log.csv compatible con EKF y visualizadores.
"""

import os
import sys
import time
import json
import struct
import math
import argparse
import threading
from typing import Optional, Dict, Any, Generator, List, Tuple

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")


def list_available_ports() -> List[Tuple[str, str]]:
    """Retorna una lista de tuplas (puerto, descripcion) para todos los puertos COM disponibles."""
    if serial is None:
        return []
    ports = serial.tools.list_ports.comports()
    return [(p.device, p.description) for p in ports]


def calculate_crc8(data: bytes) -> int:
    """Calcula CRC8 estándar (polinomio 0x07) para validación de tramas binarias opcionales."""
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


class AttitudeComplementaryFilter:
    """
    Filtro complementario de actitud (Roll, Pitch, Yaw) optimizado para 50 Hz.
    Combina la estabilidad estatica del acelerometro con la respuesta dinamica del giroscopio.
    """

    def __init__(self, alpha: float = 0.96):
        self.alpha = alpha
        self.roll_deg = 0.0
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0
        self.last_time_s = None
        self.calibrated = False

    def reset(self, initial_heading_deg: float = 0.0):
        self.roll_deg = 0.0
        self.pitch_deg = 0.0
        self.yaw_deg = float(initial_heading_deg)
        self.last_time_s = None

    def update(self, ax: float, ay: float, az: float,
               gx: float, gy: float, gz: float,
               timestamp_s: float) -> Tuple[float, float, float]:
        """
        Actualiza el estado de actitud.
        ax, ay, az en m/s^2
        gx, gy, gz en rad/s
        timestamp_s en segundos
        """
        if self.last_time_s is None:
            dt = 0.02
        else:
            dt = timestamp_s - self.last_time_s
            if dt <= 0.0 or dt > 0.5:
                dt = 0.02
        self.last_time_s = timestamp_s

        # Estimacion de alabeo (roll) y cabeceo (pitch) a partir del vector gravedad
        # Asumiendo cuerpo NED (az aprox +9.81 en reposo horizontal)
        norm_a = math.sqrt(ax * ax + ay * ay + az * az)
        if norm_a > 1e-3:
            roll_acc = math.degrees(math.atan2(ay, az))
            pitch_acc = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
        else:
            roll_acc = self.roll_deg
            pitch_acc = self.pitch_deg

        if not self.calibrated:
            self.roll_deg = roll_acc
            self.pitch_deg = pitch_acc
            self.calibrated = True

        # Integracion de giroscopo (rad/s a deg)
        roll_gyro = self.roll_deg + math.degrees(gx) * dt
        pitch_gyro = self.pitch_deg + math.degrees(gy) * dt

        # Fusion complementaria
        self.roll_deg = self.alpha * roll_gyro + (1.0 - self.alpha) * roll_acc
        self.pitch_deg = self.alpha * pitch_gyro + (1.0 - self.alpha) * pitch_acc

        # Rumbo (Yaw): Integracion pura del eje Z (con modulo 0-360 deg)
        self.yaw_deg = (self.yaw_deg + math.degrees(gz) * dt) % 360.0

        return self.roll_deg, self.pitch_deg, self.yaw_deg


class ESP32IMUReader:
    """
    Gestor de conexion serie y decodificacion de telemetria desde ESP32 con MPU6050.
    """

    BINARY_HEADER = b"\xAA\x55"
    TOTAL_BINARY_PACKET_SIZE = 31

    def __init__(self, port: str = "auto", baudrate: int = 115200, timeout: float = 0.5):
        self.requested_port = port
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self.is_connected = False
        self.filter = AttitudeComplementaryFilter(alpha=0.96)
        self.t0_ms = None
        self.packet_count = 0

    def find_esp32_port(self) -> Optional[str]:
        """Intenta identificar automaticamente el puerto COM del ESP32 o adaptador Bluetooth."""
        ports = list_available_ports()
        if not ports:
            return None

        # Priorizar puertos con nombres comunes de chips USB-UART o Bluetooth
        keywords = ["cp210", "ch340", "ftdi", "usb serial", "esp32", "bluetooth", "avion"]
        for p, desc in ports:
            d_lower = desc.lower()
            if any(k in d_lower for k in keywords):
                return p

        # Si solo hay un puerto COM disponible, seleccionarlo por defecto
        if len(ports) == 1:
            return ports[0][0]

        # Retornar el primer puerto encontrado
        return ports[0][0]

    def connect(self) -> bool:
        """Abre la conexion con el microcontrolador."""
        if serial is None:
            print("[ERROR] El modulo 'pyserial' no esta disponible. Instala con: pip install pyserial")
            return False

        if self.requested_port == "auto":
            detected = self.find_esp32_port()
            if not detected:
                ports = list_available_ports()
                print("\n[ERROR] No se detecto automaticamente ningun puerto serie.")
                if ports:
                    print("Puertos disponibles en el sistema:")
                    for dev, desc in ports:
                        print(f"  - {dev}: {desc}")
                else:
                    print("  Ningun dispositivo COM conectado actualmente.")
                return False
            self.port = detected

        try:
            print(f"Conectando a {self.port} a {self.baudrate} baudios (ESP32 MPU6050)...")
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(1.2)  # Permitir estabilizacion de DTR / RTS
            self.ser.reset_input_buffer()
            self.is_connected = True
            print(f"  [OK] Conectado exitosamente a {self.port}.")
            return True
        except Exception as e:
            print(f"[ERROR] No fue posible conectar al puerto {self.port}: {e}")
            self.is_connected = False
            return False

    def disconnect(self) -> None:
        """Cierra el puerto serie de forma segura."""
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
            self.is_connected = False
            print(f"Puerto {self.port} cerrado.")

    def read_packet(self) -> Optional[Dict[str, Any]]:
        """
        Lee una linea del puerto serie y la decodifica.
        Ignora lineas de depuracion que inicien con '#' (como los mensajes de setup del ESP32).
        """
        if not self.is_connected or not self.ser:
            return None

        try:
            raw_line = self.ser.readline()
        except Exception:
            return None

        if not raw_line:
            return None

        # 1. Intentar decodificar como texto UTF-8 / JSON
        try:
            text = raw_line.decode("utf-8", errors="ignore").strip()
            # Ignorar comentarios o mensajes de inicio del ESP32
            if text.startswith("#") or not text:
                return None

            if text.startswith("{") and text.endswith("}"):
                data = json.loads(text)
                t_ms = int(data.get("t", 0))
                if self.t0_ms is None:
                    self.t0_ms = t_ms

                sim_time_sec = (t_ms - self.t0_ms) / 1000.0 if t_ms >= self.t0_ms else float(self.packet_count) * 0.02

                ax = float(data.get("ax", 0.0))
                ay = float(data.get("ay", 0.0))
                az = float(data.get("az", 9.81))
                gx = float(data.get("gx", 0.0))
                gy = float(data.get("gy", 0.0))
                gz = float(data.get("gz", 0.0))

                roll_deg, pitch_deg, yaw_deg = self.filter.update(ax, ay, az, gx, gy, gz, sim_time_sec)
                self.packet_count += 1

                return {
                    "timestamp_ms": t_ms,
                    "sim_time_sec": round(sim_time_sec, 4),
                    "accel_x_mps2": ax,
                    "accel_y_mps2": ay,
                    "accel_z_mps2": az,
                    "gyro_p_rad_s": gx,
                    "gyro_q_rad_s": gy,
                    "gyro_r_rad_s": gz,
                    "roll_deg": round(roll_deg, 2),
                    "pitch_deg": round(pitch_deg, 2),
                    "yaw_deg": round(yaw_deg, 2),
                }
        except Exception:
            pass

        # 2. Intentar decodificar como trama binaria (si se usa formato compacto)
        if len(raw_line) >= self.TOTAL_BINARY_PACKET_SIZE and raw_line.startswith(self.BINARY_HEADER):
            payload = raw_line[2:30]
            crc_received = raw_line[30]
            if calculate_crc8(payload) == crc_received:
                t_ms, ax, ay, az, gx, gy, gz = struct.unpack("<Iffffff", payload)
                if self.t0_ms is None:
                    self.t0_ms = t_ms
                sim_time_sec = (t_ms - self.t0_ms) / 1000.0

                roll_deg, pitch_deg, yaw_deg = self.filter.update(ax, ay, az, gx, gy, gz, sim_time_sec)
                self.packet_count += 1

                return {
                    "timestamp_ms": t_ms,
                    "sim_time_sec": round(sim_time_sec, 4),
                    "accel_x_mps2": ax,
                    "accel_y_mps2": ay,
                    "accel_z_mps2": az,
                    "gyro_p_rad_s": gx,
                    "gyro_q_rad_s": gy,
                    "gyro_r_rad_s": gz,
                    "roll_deg": round(roll_deg, 2),
                    "pitch_deg": round(pitch_deg, 2),
                    "yaw_deg": round(yaw_deg, 2),
                }

        return None


# Alias de compatibilidad hacia atras
SerialIMUReader = ESP32IMUReader
STM32IMUReader = ESP32IMUReader


def simulate_esp32_stream(duration_sec: float = 30.0) -> Generator[Dict[str, Any], None, None]:
    """
    Generador de emulacion en tiempo real (50 Hz / 20 ms).
    Simula un ESP32 + MPU6050 realizando suaves movimientos de alabeo y cabeceo.
    """
    filt = AttitudeComplementaryFilter(alpha=0.96)
    start_time = time.time()
    t_ms = 0
    dt = 0.02  # 50 Hz

    while time.time() - start_time < duration_sec:
        t_sec = time.time() - start_time
        t_ms = int(t_sec * 1000)

        # Generar perfiles de movimiento suaves para demostracion
        target_roll_rad = math.radians(15.0 * math.sin(t_sec * 0.8))
        target_pitch_rad = math.radians(6.0 * math.cos(t_sec * 0.5))
        target_yaw_rate_rad = math.radians(4.0)

        gx = 15.0 * 0.8 * math.cos(t_sec * 0.8) * (math.pi / 180.0)
        gy = -6.0 * 0.5 * math.sin(t_sec * 0.5) * (math.pi / 180.0)
        gz = target_yaw_rate_rad

        g = 9.81
        ax = -g * math.sin(target_pitch_rad) + np_noise(0.04)
        ay = g * math.sin(target_roll_rad) * math.cos(target_pitch_rad) + np_noise(0.04)
        az = g * math.cos(target_roll_rad) * math.cos(target_pitch_rad) + np_noise(0.04)

        roll_deg, pitch_deg, yaw_deg = filt.update(ax, ay, az, gx, gy, gz, t_sec)

        yield {
            "timestamp_ms": t_ms,
            "sim_time_sec": round(t_sec, 4),
            "accel_x_mps2": round(ax, 3),
            "accel_y_mps2": round(ay, 3),
            "accel_z_mps2": round(az, 3),
            "gyro_p_rad_s": round(gx, 4),
            "gyro_q_rad_s": round(gy, 4),
            "gyro_r_rad_s": round(gz, 4),
            "roll_deg": round(roll_deg, 2),
            "pitch_deg": round(pitch_deg, 2),
            "yaw_deg": round(yaw_deg, 2),
        }
        time.sleep(dt)


def np_noise(scale: float) -> float:
    import random
    return random.gauss(0, scale)


def main():
    parser = argparse.ArgumentParser(description="Receptor serie para ESP32 + MPU6050 (USB / Bluetooth Serial)")
    parser.add_argument("--port", type=str, default="auto", help="Puerto COM (ej. COM3, COM4, o 'auto')")
    parser.add_argument("--baud", type=int, default=115200, help="Velocidad en baudios (default: 115200)")
    parser.add_argument("--list", action="store_true", help="Listar puertos COM disponibles en Windows y salir")
    parser.add_argument("--simulate", action="store_true", help="Modo emulador 50 Hz sin hardware fisico")
    parser.add_argument("--duration", type=float, default=60.0, help="Duracion de captura en segundos (default: 60s)")
    args = parser.parse_args()

    if args.list:
        ports = list_available_ports()
        print("\n=======================================================")
        print("  PUERTOS SERIE COM DISPONIBLES EN ESTE EQUIPO")
        print("=======================================================")
        if ports:
            for dev, desc in ports:
                print(f"  * {dev:<8} : {desc}")
        else:
            print("  [AVISO] No se encontraron puertos COM activos.")
            print("  Conecta el ESP32 por USB o vincula el Bluetooth 'AvionESP32'.")
        print("=======================================================\n")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    out_csv = os.path.join(DATA_DIR, "physical_imu_log.csv")

    packets = []
    start_wall = time.time()

    if args.simulate:
        print("\n=======================================================")
        print("  MODO EMULADOR ESP32 ACTIVADO (50 Hz)")
        print("=======================================================")
        print(f"Generando flujo sintetico durante {args.duration:.1f} s...")
        print("Formato: [Tiempo] | Aceleracion (X, Y, Z) | Giroscopo (P, Q, R) | Actitud (Roll, Pitch, Yaw)\n")

        for p in simulate_esp32_stream(duration_sec=args.duration):
            packets.append(p)
            if len(packets) % 25 == 0:
                print(f"  t={p['sim_time_sec']:5.1f}s | Acc: [{p['accel_x_mps2']:+5.2f}, {p['accel_y_mps2']:+5.2f}, {p['accel_z_mps2']:+5.2f}] m/s² | "
                      f"Actitud: Roll={p['roll_deg']:+5.1f}° Pitch={p['pitch_deg']:+5.1f}° Yaw={p['yaw_deg']:05.1f}°")
    else:
        reader = ESP32IMUReader(port=args.port, baudrate=args.baud)
        if not reader.connect():
            print("\n[SUGERENCIA] Puedes probar el sistema sin conectar el hardware usando:")
            print("  python src/serial_imu_reader.py --simulate")
            print("O listar puertos con:")
            print("  python src/serial_imu_reader.py --list")
            return

        print("\n=======================================================")
        print("  TELEMETRIA EN VIVO DESDE ESP32 + MPU6050 (50 Hz)")
        print("=======================================================")
        print("Presiona Ctrl+C para detener y guardar los datos.\n")

        try:
            while time.time() - start_wall < args.duration:
                packet = reader.read_packet()
                if packet:
                    packets.append(packet)
                    if len(packets) % 25 == 0:
                        hz = len(packets) / max(time.time() - start_wall, 1e-3)
                        print(f"  [{hz:4.1f} Hz] t={packet['sim_time_sec']:5.1f}s | "
                              f"Roll: {packet['roll_deg']:+5.1f}° | Pitch: {packet['pitch_deg']:+5.1f}° | Yaw: {packet['yaw_deg']:05.1f}° | "
                              f"Az: {packet['accel_z_mps2']:+5.2f} m/s²")
        except KeyboardInterrupt:
            print("\nCaptura detenida por el usuario.")
        finally:
            reader.disconnect()

    if packets:
        import pandas as pd
        df = pd.DataFrame(packets)
        df.to_csv(out_csv, index=False)
        total_time = time.time() - start_wall
        mean_hz = len(packets) / total_time
        print(f"\n[OK] {len(packets)} muestras registradas en: {out_csv}")
        print(f"     Tasa efectiva promedio: {mean_hz:.1f} Hz")


if __name__ == "__main__":
    main()
