"""Capture frames from an ESP32-CAM MJPEG stream and synchronize available telemetry.

Example (project root):
  python src/live_camera_capture.py --stream-url http://192.168.4.1:81/stream --port COM3

Frames are saved as JPEGs in data/live_capture/frames and indexed in
data/live_capture/frames.csv. A trained segmentation model is not required for
capture; inference can be connected after the real-image model is trained.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.serial_imu_reader import STM32IMUReader


CSV_FIELDS = [
    "frame_index", "image_path", "image_filename", "capture_time_utc", "timestamp_iso", "host_monotonic_ns",
    "segmentation_mask_path", "segmentation_inference_ms",
    "sim_time_sec", "imu_timestamp_ms", "imu_age_ms",
    "accel_x_mps2", "accel_y_mps2", "accel_z_mps2",
    "gyro_p_rad_s", "gyro_q_rad_s", "gyro_r_rad_s",
    "latitude_deg", "longitude_deg", "lat_geod_deg", "lon_geod_deg", "altitude_agl_m",
    "roll_deg", "pitch_deg", "yaw_deg", "reference_map_meta",
]


class OnnxSegmenter:
    """Optional ONNX semantic-segmentation runner for exported NCHW logits models."""

    def __init__(self, model_path: str, input_size: int = 256) -> None:
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.input_size = input_size
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def infer(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = frame.shape[:2]
        resized = cv2.resize(frame, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = ((rgb - self.mean) / self.std).transpose(2, 0, 1)[None, ...]
        start = time.perf_counter()
        self.net.setInput(blob)
        logits = self.net.forward()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if logits.ndim != 4 or logits.shape[0] != 1:
            raise ValueError(f"El ONNX debe producir logits 1xCxHxW o 1xHxWxC; forma recibida: {logits.shape}")
        if logits.shape[1] <= logits.shape[-1]:
            scores = logits[0]  # C x H x W
            mask = np.argmax(scores, axis=0)
        else:
            scores = logits[0]  # H x W x C
            mask = np.argmax(scores, axis=2)
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        return mask, elapsed_ms


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Deterministic palette for optional preview overlay."""
    palette = np.zeros((256, 3), dtype=np.uint8)
    for class_id in range(256):
        # BGR colors; avoid very dark colors that disappear in the overlay.
        palette[class_id] = ((37 * class_id + 60) % 220 + 25,
                             (83 * class_id + 80) % 220 + 25,
                             (131 * class_id + 100) % 220 + 25)
    return palette[mask]


class LatestTelemetry:
    """Thread-safe latest-packet store; camera and serial have independent rates."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.packet: Optional[Dict[str, Any]] = None
        self.received_monotonic_ns: Optional[int] = None

    def update(self, packet: Dict[str, Any]) -> None:
        with self.lock:
            self.packet = packet
            self.received_monotonic_ns = time.monotonic_ns()

    def snapshot(self) -> tuple[Optional[Dict[str, Any]], Optional[int]]:
        with self.lock:
            return self.packet, self.received_monotonic_ns


def read_telemetry(reader: STM32IMUReader, latest: LatestTelemetry, stop: threading.Event) -> None:
    while not stop.is_set():
        packet = reader.read_packet()
        if packet:
            latest.update(packet)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recibe fotogramas JPEG/MJPEG de ESP32-CAM y los guarda con hora y telemetría disponible."
    )
    parser.add_argument("--stream-url", required=True, help="URL MJPEG de la ESP32-CAM, por ejemplo http://192.168.4.1:81/stream")
    parser.add_argument("--port", help="Puerto serie del STM32/IMU (ej. COM3). Omítelo para probar solo la cámara.")
    parser.add_argument("--baud", type=int, default=115200, help="Baudios del STM32; por defecto 115200.")
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "live_capture"), help="Carpeta de salida de esta captura.")
    parser.add_argument("--duration", type=float, default=0.0, help="Duración en segundos; 0 = hasta pulsar q o Ctrl+C.")
    parser.add_argument("--save-every", type=int, default=1, help="Guardar cada N fotogramas; por defecto guarda todos.")
    parser.add_argument("--headless", action="store_true", help="No abre ventana de video; útil en Jetson sin pantalla.")
    parser.add_argument("--serial-timeout", type=float, default=0.15, help="Timeout de lectura serie en segundos.")
    parser.add_argument("--map-meta", help="JSON de metadatos del mapa real; queda anotado en el CSV.")
    parser.add_argument("--onnx-model", help="Modelo de segmentación ONNX opcional; si se omite solo se capturan imágenes.")
    parser.add_argument("--seg-input-size", type=int, default=256, help="Tamaño cuadrado de entrada del ONNX.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.save_every < 1:
        raise ValueError("--save-every debe ser mayor que cero")
    if args.duration < 0:
        raise ValueError("--duration no puede ser negativo")

    output_dir = Path(args.output_dir).expanduser().resolve()
    frames_dir = output_dir / "frames"
    masks_dir = output_dir / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if args.onnx_model:
        masks_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "frames.csv"

    segmenter = OnnxSegmenter(args.onnx_model, args.seg_input_size) if args.onnx_model else None

    camera = cv2.VideoCapture(args.stream_url)
    if not camera.isOpened():
        raise RuntimeError(
            f"No se pudo abrir el flujo {args.stream_url}. Comprueba primero la URL en un navegador y que PC/Jetson y cámara estén en la misma red."
        )
    try:
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    reader = None
    latest = LatestTelemetry()
    stop_telemetry = threading.Event()
    telemetry_thread = None
    if args.port:
        reader = STM32IMUReader(port=args.port, baudrate=args.baud, timeout=args.serial_timeout)
        if not reader.connect():
            camera.release()
            raise RuntimeError(f"No se pudo abrir el puerto serie {args.port}")
        telemetry_thread = threading.Thread(
            target=read_telemetry, args=(reader, latest, stop_telemetry), daemon=True
        )
        telemetry_thread.start()
        print(f"Telemetría serie activa en {args.port} a {args.baud} baudios.")
    else:
        print("Modo cámara solamente: no se indicó --port; campos IMU/GPS quedarán vacíos.")

    start_ns = time.monotonic_ns()
    frame_index = 0
    saved_count = 0
    last_report = time.monotonic()
    print(f"Recibiendo {args.stream_url}")
    print(f"Fotogramas y manifiesto: {output_dir}")
    if args.map_meta:
        print(f"Mapa asociado: {args.map_meta}")
    print("La captura guarda imágenes sin segmentar. Pulsa q para terminar.")
    if segmenter:
        print(f"Segmentación ONNX activa: {args.onnx_model}")
    else:
        print("No se indicó --onnx-model; segmentación desactivada hasta tener un modelo exportado.")

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()

            while True:
                ok, frame = camera.read()
                capture_ns = time.monotonic_ns()
                if not ok or frame is None:
                    print("Se perdió el flujo de cámara; terminando captura.")
                    break

                elapsed_sec = (capture_ns - start_ns) / 1e9
                should_save = frame_index % args.save_every == 0
                if should_save:
                    capture_time = utc_now_iso()
                    stamp = capture_time.replace("-", "").replace(":", "").replace(".", "")
                    image_name = f"frame_{frame_index:08d}_{stamp}.jpg"
                    image_path = frames_dir / image_name
                    if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise RuntimeError(f"No se pudo guardar el fotograma: {image_path}")

                    mask_rel_path = ""
                    inference_ms = ""
                    if segmenter:
                        mask, inference_ms = segmenter.infer(frame)
                        mask_path = masks_dir / f"{Path(image_name).stem}_mask.png"
                        if not cv2.imwrite(str(mask_path), mask):
                            raise RuntimeError(f"No se pudo guardar la máscara: {mask_path}")
                        mask_rel_path = os.path.relpath(mask_path, output_dir)

                    packet, imu_received_ns = latest.snapshot()
                    imu_age_ms = ""
                    if imu_received_ns is not None:
                        imu_age_ms = round(max(0, capture_ns - imu_received_ns) / 1e6, 3)
                    row: Dict[str, Any] = {
                        "frame_index": frame_index,
                        "image_path": os.path.relpath(image_path, output_dir),
                        "image_filename": image_name,
                        "capture_time_utc": capture_time,
                        "timestamp_iso": capture_time,
                        "host_monotonic_ns": capture_ns,
                        "segmentation_mask_path": mask_rel_path,
                        "segmentation_inference_ms": inference_ms,
                        "sim_time_sec": round(elapsed_sec, 6),
                        "imu_age_ms": imu_age_ms,
                        "reference_map_meta": args.map_meta or "",
                    }
                    if packet:
                        row["imu_timestamp_ms"] = packet.get("timestamp_ms", "")
                        for key in CSV_FIELDS:
                            if key in packet and key != "timestamp_ms":
                                row[key] = packet[key]
                        row["lat_geod_deg"] = packet.get("latitude_deg", "")
                        row["lon_geod_deg"] = packet.get("longitude_deg", "")
                    writer.writerow(row)
                    csv_file.flush()
                    saved_count += 1

                if not args.headless:
                    preview = frame.copy()
                    if segmenter and should_save:
                        # Display the most recently saved frame's segmentation overlay.
                        color_mask = colorize_mask(mask)
                        preview = cv2.addWeighted(preview, 0.62, color_mask, 0.38, 0.0)
                    status = "SEGMENTANDO" if segmenter else "CAPTURA - SIN MODELO"
                    cv2.putText(preview, f"{status} {elapsed_sec:.1f}s | {frame.shape[1]}x{frame.shape[0]}",
                                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    cv2.imshow("ESP32-CAM - terreno", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_index += 1
                if time.monotonic() - last_report >= 2.0:
                    print(f"Fotogramas recibidos: {frame_index}; guardados: {saved_count}; tasa media: {frame_index / max(elapsed_sec, 1e-6):.1f} FPS")
                    last_report = time.monotonic()
                if args.duration and elapsed_sec >= args.duration:
                    break
    except KeyboardInterrupt:
        print("Captura interrumpida por el usuario.")
    finally:
        stop_telemetry.set()
        if telemetry_thread is not None:
            telemetry_thread.join(timeout=1.5)
        if reader is not None:
            reader.disconnect()
        camera.release()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"Captura terminada: {saved_count} imágenes en {frames_dir}")
    print(f"Manifiesto CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
