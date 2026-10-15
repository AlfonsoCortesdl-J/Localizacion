"""
serial_imu_reader.py
====================
Controlador y receptor serie en tiempo real para adquirir lecturas de una IMU física
(ej. MPU-6050, ICM-42688P, BNO085) conectada a un microcontrolador STM32.

Características:
----------------
1. Autodetección de puertos COM en Windows.
2. Parser dual de tramas:
   - Formato JSON / ASCII liviano:
     {"t":12345,"ax":0.05,"ay":-0.02,"az":-9.81,"gx":0.01,"gy":-0.01,"gz":0.02}
   - Formato Binario empaquetado (31 bytes):
     [0xAA, 0x55] + [uint32 timestamp_ms] + [3x float32 accel] + [3x float32 gyro] + [CRC8]
3. Modo de Emulación / Loopback integrado (--simulate):
   Permite probar la interfaz y el flujo de datos sin necesidad de conectar hardware físico.
4. Registro en tiempo real en data/physical_imu_log.csv.

Ejemplo de Firmware C para STM32 (STM32CubeIDE / HAL):
-------------------------------------------------------
Ver la sección de documentación al final del archivo con el código C listo para flashear.
"""

import os
import sys
import time
import json
import struct
import argparse
from typing import Optional, Dict, Any, Generator

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


def list_available_ports() -> list:
    """Lista todos los puertos COM disponibles en Windows."""
    if serial is None:
        print("[AVISO] pyserial no está instalado. Ejecuta: pip install pyserial")
        return []
    ports = serial.tools.list_ports.comports()
    return [p.device for p in ports]


def calculate_crc8(data: bytes) -> int:
    """Calcula CRC8 estándar (polinomio 0x07) para validación de tramas binarias."""
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


class STM32IMUReader:
    """
    Gestor de comunicación serie con STM32 para telemetría inercial.
    """

    BINARY_HEADER = b"\xAA\x55"
    BINARY_PAYLOAD_SIZE = 28  # 4 bytes uint32 + 3*4 float accel + 3*4 float gyro
    TOTAL_BINARY_PACKET_SIZE = 2 + 28 + 1  # 31 bytes

    def __init__(self, port: str = "COM3", baudrate: int = 115200, timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self.is_connected = False

    def connect(self) -> bool:
        """Abre la conexión con el puerto serie."""
        if serial is None:
            raise RuntimeError("pyserial no está disponible.")
        try:
            print(f"Abriendo puerto {self.port} a {self.baudrate} baudios...")
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(1.5)  # Esperar reinicio de microcontrolador por DTR
            self.ser.reset_input_buffer()
            self.is_connected = True
            print(f"  [OK] Conectado exitosamente a {self.port}.")
            return True
        except Exception as e:
            print(f"[ERROR] No se pudo conectar a {self.port}: {e}")
            self.is_connected = False
            return False

    def disconnect(self) -> None:
        """Cierra el puerto serie."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.is_connected = False
            print(f"Puerto {self.port} cerrado.")

    def read_packet(self) -> Optional[Dict[str, Any]]:
        """
        Lee y decodifica un paquete de la IMU (JSON o Binario).
        """
        if not self.is_connected or not self.ser:
            return None

        # Intentar leer línea ASCII/JSON primero
        line = self.ser.readline()
        if not line:
            return None

        # 1. Intentar decodificar como JSON
        try:
            text = line.decode("utf-8", errors="ignore").strip()
            if text.startswith("{") and text.endswith("}"):
                data = json.loads(text)
                return {
                    "timestamp_ms": data.get("t", 0),
                    "accel_x_mps2": float(data.get("ax", 0.0)),
                    "accel_y_mps2": float(data.get("ay", 0.0)),
                    "accel_z_mps2": float(data.get("az", 0.0)),
                    "gyro_p_rad_s": float(data.get("gx", 0.0)),
                    "gyro_q_rad_s": float(data.get("gy", 0.0)),
                    "gyro_r_rad_s": float(data.get("gz", 0.0)),
                }
        except Exception:
            pass

        # 2. Intentar decodificar como trama binaria estructurada
        if len(line) >= self.TOTAL_BINARY_PACKET_SIZE and line.startswith(self.BINARY_HEADER):
            payload = line[2:30]
            crc_received = line[30]
            if calculate_crc8(payload) == crc_received:
                t_ms, ax, ay, az, gx, gy, gz = struct.unpack("<Iffffff", payload)
                return {
                    "timestamp_ms": t_ms,
                    "accel_x_mps2": ax,
                    "accel_y_mps2": ay,
                    "accel_z_mps2": az,
                    "gyro_p_rad_s": gx,
                    "gyro_q_rad_s": gy,
                    "gyro_r_rad_s": gz,
                }

        return None


def simulate_stm32_stream(csv_source_path: str) -> Generator[Dict[str, Any], None, None]:
    """
    Generador de emulación en tiempo real: simula el microcontrolador STM32
    transmitiendo a 100 Hz las lecturas inerciales.
    """
    import pandas as pd
    df = pd.read_csv(csv_source_path)
    print(f"Iniciando emulador STM32 con {len(df)} muestras a 100 Hz...")

    for i in range(len(df)):
        row = df.iloc[i]
        yield {
            "timestamp_ms": int(row["sim_time_sec"] * 1000.0),
            "accel_x_mps2": float(row["accel_x_mps2"]),
            "accel_y_mps2": float(row["accel_y_mps2"]),
            "accel_z_mps2": float(row["accel_z_mps2"]),
            "gyro_p_rad_s": float(row["gyro_p_rad_s"]),
            "gyro_q_rad_s": float(row["gyro_q_rad_s"]),
            "gyro_r_rad_s": float(row["gyro_r_rad_s"]),
        }
        time.sleep(0.01)  # 100 Hz


def main():
    parser = argparse.ArgumentParser(description="Receptor serie para STM32 + IMU física")
    parser.add_argument("--port", type=str, default="COM3", help="Puerto COM serie (ej. COM3, COM4)")
    parser.add_argument("--baud", type=int, default=115200, help="Velocidad en baudios (115200 o 921600)")
    parser.add_argument("--simulate", action="store_true", help="Modo emulador sin hardware físico")
    parser.add_argument("--duration", type=float, default=10.0, help="Duración de la captura en segundos")
    args = parser.parse_args()

    data_dir = os.path.join(parent_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    out_csv = os.path.join(data_dir, "physical_imu_log.csv")

    packets = []
    start_time = time.time()

    if args.simulate:
        print("\n=== MODO EMULADOR STM32 ACTIVADO ===")
        print(f"Transmitiendo flujo IMU emulado durante {args.duration:.1f} s...")
        source_csv = os.path.join(data_dir, "imu_measurements.csv")
        if not os.path.exists(source_csv):
            print(f"[ERROR] No se encontró {source_csv}. Ejecuta primero record_flight.py.")
            return

        stream = simulate_stm32_stream(source_csv)
        packet_count = 0

        for packet in stream:
            packets.append(packet)
            packet_count += 1
            if packet_count % 50 == 0:
                print(f"  [STM32 EMULADO] t={packet['timestamp_ms']:6d} ms | Accel: [{packet['accel_x_mps2']:+6.2f}, {packet['accel_y_mps2']:+6.2f}, {packet['accel_z_mps2']:+6.2f}] m/s² | Gyro: [{packet['gyro_p_rad_s']:+6.3f}, {packet['gyro_q_rad_s']:+6.3f}, {packet['gyro_r_rad_s']:+6.3f}] rad/s")

            if time.time() - start_time >= args.duration:
                break
    else:
        print("\n=== MODO HARDWARE FÍSICO STM32 ===")
        available_ports = list_available_ports()
        print(f"Puertos COM detectados en el sistema: {available_ports}")

        reader = STM32IMUReader(port=args.port, baudrate=args.baud)
        if not reader.connect():
            print(f"\n[SUGERENCIA] Si aún no has conectado el STM32, ejecuta con --simulate:")
            print(f"  python src/serial_imu_reader.py --simulate --duration 10")
            return

        print(f"Escuchando tramas serie en {args.port} durante {args.duration:.1f} s...")
        try:
            while time.time() - start_time < args.duration:
                packet = reader.read_packet()
                if packet:
                    packets.append(packet)
                    if len(packets) % 25 == 0:
                        print(f"  [STM32 HARDWARE] t={packet['timestamp_ms']:6d} ms | Accel: [{packet['accel_x_mps2']:+6.2f}, {packet['accel_y_mps2']:+6.2f}, {packet['accel_z_mps2']:+6.2f}] m/s²")
        finally:
            reader.disconnect()

    if packets:
        import pandas as pd
        df = pd.DataFrame(packets)
        df.to_csv(out_csv, index=False)
        rate = len(packets) / (time.time() - start_time)
        print(f"\n[Archivo guardado] {len(packets)} muestras guardadas en: {out_csv} ({rate:.1f} Hz medio)")


# ==============================================================================
# CÓDIGO DE EJEMPLO PARA STM32CubeIDE (HAL C) — MPU6050 + UART
# ==============================================================================
STM32_C_EXAMPLE = """
/*
 * Ejemplo de código para STM32F4 / STM32G4 / STM32F103 (STM32CubeIDE)
 * Transmisión de tramas IMU a 100 Hz por UART hacia Python
 */

#include "main.h"
#include <stdio.h>
#include <string.h>

extern UART_HandleTypeDef huart2;
extern I2C_HandleTypeDef hi2c1;

#define MPU6050_ADDR (0x68 << 1)

void MPU6050_Init(void) {
    uint8_t check, data;
    HAL_I2C_Mem_Read(&hi2c1, MPU6050_ADDR, 0x75, 1, &check, 1, 100);
    if (check == 0x68) {
        data = 0x00; // Salir de modo sleep
        HAL_I2C_Mem_Write(&hi2c1, MPU6050_ADDR, 0x6B, 1, &data, 1, 100);
        data = 0x08; // Rango acelerómetro +/- 4g
        HAL_I2C_Mem_Write(&hi2c1, MPU6050_ADDR, 0x1C, 1, &data, 1, 100);
        data = 0x08; // Rango giróscopo +/- 500 dps
        HAL_I2C_Mem_Write(&hi2c1, MPU6050_ADDR, 0x1B, 1, &data, 1, 100);
    }
}

void IMU_Transmit_JSON(void) {
    uint8_t raw[14];
    HAL_I2C_Mem_Read(&hi2c1, MPU6050_ADDR, 0x3B, 1, raw, 14, 100);

    int16_t ax_raw = (raw[0] << 8) | raw[1];
    int16_t ay_raw = (raw[2] << 8) | raw[3];
    int16_t az_raw = (raw[4] << 8) | raw[5];
    int16_t gx_raw = (raw[8] << 8) | raw[9];
    int16_t gy_raw = (raw[10] << 8) | raw[11];
    int16_t gz_raw = (raw[12] << 8) | raw[13];

    // Conversión a unidades del Sistema Internacional (m/s^2 y rad/s)
    float ax = (ax_raw / 8192.0f) * 9.80665f;
    float ay = (ay_raw / 8192.0f) * 9.80665f;
    float az = (az_raw / 8192.0f) * 9.80665f;

    float gx = (gx_raw / 65.5f) * (3.14159265f / 180.0f);
    float gy = (gy_raw / 65.5f) * (3.14159265f / 180.0f);
    float gz = (gz_raw / 65.5f) * (3.14159265f / 180.0f);

    uint32_t t = HAL_GetTick();
    char buf[128];
    int len = snprintf(buf, sizeof(buf),
                       "{\\"t\\":%lu,\\"ax\\":%.3f,\\"ay\\":%.3f,\\"az\\":%.3f,\\"gx\\":%.4f,\\"gy\\":%.4f,\\"gz\\":%.4f}\\r\\n",
                       t, ax, ay, az, gx, gy, gz);
    HAL_UART_Transmit(&huart2, (uint8_t*)buf, len, 50);
}
"""

if __name__ == "__main__":
    main()
