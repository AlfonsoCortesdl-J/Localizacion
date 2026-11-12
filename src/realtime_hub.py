"""
realtime_hub.py
===============
Hub unificado en tiempo real para el proyecto de navegación aérea.

Un solo proceso sirve:

* Panel web (``/``) con escena 3D, cámaras y telemetría.
* API REST (``/api/...``) para estado, escena, control y telemetría.
* WebSocket (puerto aparte) que empuja el estado a ~20 Hz.
* Cámaras MJPEG (``/camera/nadir.mjpg``, ``sky``, ``real``).
* Ingesta de telemetría externa por HTTP POST y UDP.
* Grabación de sesiones en ``data/live_sessions/``.

Fuentes de datos
----------------
``sim``     Reproduce el vuelo del Cessna 172 (flight_log + fused + VO).
``campus``  Reproduce la misión de inspección del campus (campus_mission.json).
``serial``  Lee una IMU física ESP32/STM32 por puerto serie.
``live``    Recibe telemetría de cualquier cliente (HTTP/UDP) y la muestra.
``replay``  Reproduce una sesión grabada.

Uso típico
----------
    python src/realtime_hub.py                       # Cessna simulado
    python src/realtime_hub.py --mission campus      # Dron del campus
    python src/realtime_hub.py --source serial --port COM3
    python src/realtime_hub.py --source live         # espera POST/UDP
    python src/realtime_hub.py --host 0.0.0.0        # accesible en LAN

El panel queda en http://localhost:8080 y el WebSocket en ws://localhost:8081.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import mimetypes
import os
import socket
import sys
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
WEB_DIR = ROOT / "src" / "web"
SESSIONS_DIR = DATA_DIR / "live_sessions"
MISSION_PATH = DATA_DIR / "campus_mission.json"
SCENE_PATH = DATA_DIR / "campus_scene.json"

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import websockets
except Exception:  # pragma: no cover
    websockets = None  # type: ignore

try:
    from src.serial_imu_reader import ESP32IMUReader, list_available_ports
except Exception:  # pragma: no cover
    try:
        from serial_imu_reader import ESP32IMUReader, list_available_ports  # type: ignore
    except Exception:
        ESP32IMUReader = None  # type: ignore
        list_available_ports = lambda: []  # type: ignore


GRAVITY = 9.80665
BASE_ALT_CESSNA_M = 380.0


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Tipo no serializable: {type(value)!r}")


def project_geodetic(lat: float, lon: float, center_lat: float, center_lon: float,
                     k_lat: float = 111320.0, k_lon: Optional[float] = None) -> Tuple[float, float]:
    """Devuelve (x=Este, z=-Norte) en metros locales."""
    if k_lon is None:
        k_lon = k_lat * math.cos(math.radians(center_lat))
    return (lon - center_lon) * k_lon, -(lat - center_lat) * k_lat


def rotation_body_to_ned(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Matriz de rotación cuerpo (FRD) a NED, convención Z-Y-X."""
    phi, theta, psi = np.radians([roll_deg, pitch_deg, yaw_deg])
    c_ph, s_ph = math.cos(phi), math.sin(phi)
    c_th, s_th = math.cos(theta), math.sin(theta)
    c_ps, s_ps = math.cos(psi), math.sin(psi)
    return np.array(
        [
            [c_th * c_ps, s_ph * s_th * c_ps - c_ph * s_ps, c_ph * s_th * c_ps + s_ph * s_ps],
            [c_th * s_ps, s_ph * s_th * s_ps + c_ph * c_ps, c_ph * s_th * s_ps - s_ph * c_ps],
            [-s_th, s_ph * c_th, c_ph * c_th],
        ],
        dtype=float,
    )


def _nearest_index(times: np.ndarray, t: float) -> int:
    idx = int(np.searchsorted(times, t))
    return max(0, min(len(times) - 1, idx))


# ---------------------------------------------------------------------------
# Navegación inercial para datos en vivo (sin GPS)
# ---------------------------------------------------------------------------
class DeadReckoning:
    """Integra IMU (FRD) a posición ENU local con amortiguamiento y reset."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.x = 0.0  # Este (m)
        self.y = 0.0  # Arriba (m)
        self.z = 0.0  # Sur (m)
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.last_t: Optional[float] = None
        self.max_speed = 60.0

    def update(
        self,
        accel: Tuple[float, float, float],
        gyro: Optional[Tuple[float, float, float]] = None,
        attitude: Optional[Tuple[float, float, float]] = None,
        dt: Optional[float] = None,
    ) -> None:
        now = time.monotonic()
        if dt is None:
            dt = 0.02 if self.last_t is None else max(1e-4, min(0.25, now - self.last_t))
        self.last_t = now

        if attitude is not None:
            self.roll, self.pitch, self.yaw = attitude
        elif gyro is not None:
            # Integración cinemática de Euler como respaldo.
            p, q, r = gyro
            phi, theta = math.radians(self.roll), math.radians(self.pitch)
            c_ph, s_ph = math.cos(phi), math.sin(phi)
            t_th = math.tan(max(-1.4, min(1.4, theta)))
            sec_th = 1.0 / max(0.1, math.cos(theta))
            self.roll += (p + s_ph * t_th * q + c_ph * t_th * r) * dt * 57.2957795
            self.pitch += (c_ph * q - s_ph * r) * dt * 57.2957795
            self.yaw += (s_ph * sec_th * q + c_ph * sec_th * r) * dt * 57.2957795
            self.yaw %= 360.0

        R = rotation_body_to_ned(self.roll, self.pitch, self.yaw)
        f_ned = R @ np.array(accel, dtype=float)
        a_ned = f_ned + np.array([0.0, 0.0, GRAVITY])
        # NED -> ENU local (x=Este, y=Arriba, z=Sur)
        ax = float(a_ned[1])
        ay = float(-a_ned[2])
        az = float(-a_ned[0])

        # Amortiguamiento suave: evita que el sesgo del sensor integre sin límite.
        damp = math.exp(-0.35 * dt)
        self.vx = (self.vx + ax * dt) * damp
        self.vy = (self.vy + ay * dt) * damp
        self.vz = (self.vz + az * dt) * damp
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2 + self.vz ** 2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.vx *= scale
            self.vy *= scale
            self.vz *= scale
        if speed < 0.08:
            self.vx = self.vy = self.vz = 0.0
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
        # Límite de seguridad para no salir de la escena.
        self.x = max(-1200.0, min(1200.0, self.x))
        self.z = max(-1200.0, min(1200.0, self.z))
        self.y = max(-30.0, min(400.0, self.y))


# ---------------------------------------------------------------------------
# Cámaras
# ---------------------------------------------------------------------------
class CameraFeed:
    name = "camera"

    def latest_jpeg(self) -> Optional[bytes]:
        raise NotImplementedError

    def status(self) -> Dict[str, Any]:
        return {"name": self.name, "online": self.latest_jpeg() is not None}


class SyntheticCamera(CameraFeed):
    """Genera un patrón de prueba cuando no hay cámara disponible."""

    def __init__(self, name: str, label: str, color: Tuple[int, int, int] = (30, 160, 220)):
        self.name = name
        self.label = label
        self.color = color
        self._jpeg: Optional[bytes] = None
        self._t = 0.0

    def latest_jpeg(self) -> Optional[bytes]:
        if cv2 is None:
            return None
        now = time.monotonic()
        if self._jpeg is not None and now - self._t < 0.5:
            return self._jpeg
        img = np.zeros((360, 480, 3), dtype=np.uint8)
        img[:] = (18, 24, 34)
        for y in range(0, 360, 24):
            cv2.line(img, (0, y), (480, y), (28, 38, 52), 1)
        for x in range(0, 480, 24):
            cv2.line(img, (x, 0), (x, 360), (28, 38, 52), 1)
        cv2.circle(img, (240, 180), 54, self.color, 2)
        cv2.putText(img, "SIN SENAL", (150, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 220, 240), 2)
        cv2.putText(img, self.label, (120, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 1)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        self._jpeg = buf.tobytes() if ok else None
        self._t = now
        return self._jpeg


class SimCamera(CameraFeed):
    """Fotogramas virtuales nadir/cielo indexados por tiempo de simulación."""

    def __init__(self, name: str, csv_path: Path, frames_dir: Path, fps: float):
        self.name = name
        self.csv_path = csv_path
        self.frames_dir = frames_dir
        self.fps = fps
        self._files: List[str] = []
        self._times: np.ndarray = np.array([], dtype=float)
        self._index = -1
        self._jpeg: Optional[bytes] = None
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.csv_path.exists():
            return
        try:
            df = pd.read_csv(self.csv_path)
            self._files = df["image_filename"].astype(str).tolist()
            self._times = df["sim_time_sec"].to_numpy(dtype=float)
        except Exception as exc:
            print(f"[camera] No se pudo cargar {self.csv_path}: {exc}")

    def set_time(self, sim_time: float) -> None:
        if not self._files:
            return
        idx = _nearest_index(self._times, sim_time)
        if idx == self._index:
            return
        path = self.frames_dir / self._files[idx]
        if not path.exists():
            return
        if cv2 is None:
            return
        img = cv2.imread(str(path))
        if img is None:
            return
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()
                self._index = idx

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "online": self.latest_jpeg() is not None,
            "frames": len(self._files),
            "index": self._index,
            "kind": "virtual",
        }


class MjpegUrlReader:
    """Lector MJPEG crudo (ESP32-CAM) con reconexión automática."""

    def __init__(self, url: str):
        self.url = url
        self._jpeg: Optional[bytes] = None
        self._error: Optional[str] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                req = urllib.request.Request(self.url, headers={"User-Agent": "TecMM-Hub/1.0"})
                with urllib.request.urlopen(req, timeout=8) as response:
                    buf = b""
                    while not self._stop.is_set():
                        chunk = response.read(4096)
                        if not chunk:
                            break
                        buf += chunk
                        start = buf.find(b"\xff\xd8")
                        end = buf.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                        while start >= 0 and end >= 0:
                            frame = buf[start : end + 2]
                            with self._lock:
                                self._jpeg = frame
                                self._error = None
                            buf = buf[end + 2 :]
                            start = buf.find(b"\xff\xd8")
                            end = buf.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                        if len(buf) > 4_000_000:
                            buf = buf[-1_000_000:]
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
                time.sleep(1.5)

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    def stop(self) -> None:
        self._stop.set()


class RealCamera(CameraFeed):
    """Webcam local o URL MJPEG, con reconexión y estado."""

    def __init__(self, name: str = "real"):
        self.name = name
        self._source: Any = None
        self._kind = "none"
        self._jpeg: Optional[bytes] = None
        self._error: Optional[str] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._mjpeg: Optional[MjpegUrlReader] = None
        self._fps = 0.0
        self._last_frame_t = 0.0

    def configure(self, source: Any) -> None:
        self.stop()
        self._source = source
        self._kind = "url" if isinstance(source, str) and source else ("index" if isinstance(source, int) else "none")
        self._stop = threading.Event()
        if self._kind == "url" and str(source).lower().startswith(("http://", "https://")):
            # Para ESP32-CAM se usa el lector MJPEG crudo; es más tolerante.
            self._mjpeg = MjpegUrlReader(str(source))
        else:
            self._mjpeg = None
            self._thread = threading.Thread(target=self._run_capture, daemon=True)
            self._thread.start()

    def _run_capture(self) -> None:
        if cv2 is None:
            return
        cap = None
        while not self._stop.is_set():
            try:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(self._source)
                    if isinstance(self._source, int):
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    if not cap.isOpened():
                        raise RuntimeError(f"No se pudo abrir la cámara {self._source!r}")
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Lectura de cámara vacía")
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if ok:
                    now = time.monotonic()
                    with self._lock:
                        if self._last_frame_t:
                            inst = 1.0 / max(1e-3, now - self._last_frame_t)
                            self._fps = 0.85 * self._fps + 0.15 * inst if self._fps else inst
                        self._last_frame_t = now
                        self._jpeg = buf.tobytes()
                        self._error = None
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
                if cap is not None:
                    cap.release()
                    cap = None
                time.sleep(1.2)
        if cap is not None:
            cap.release()

    def latest_jpeg(self) -> Optional[bytes]:
        if self._mjpeg is not None:
            return self._mjpeg.latest_jpeg()
        with self._lock:
            return self._jpeg

    def status(self) -> Dict[str, Any]:
        error = self._mjpeg.error() if self._mjpeg is not None else self._error
        with self._lock:
            fps = self._fps
        return {
            "name": self.name,
            "online": self.latest_jpeg() is not None,
            "kind": self._kind,
            "source": str(self._source) if self._source is not None else None,
            "fps": round(fps, 1),
            "error": error,
        }

    def stop(self) -> None:
        self._stop.set()
        if self._mjpeg is not None:
            self._mjpeg.stop()
            self._mjpeg = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None


class CameraManager:
    def __init__(self) -> None:
        self.nadir = SimCamera("nadir", DATA_DIR / "camera_ground_frames.csv", DATA_DIR / "camera_ground", 5.0)
        self.sky = SimCamera("sky", DATA_DIR / "camera_sky_frames.csv", DATA_DIR / "camera_sky", 2.0)
        self.real = RealCamera("real")
        self._synthetic = {
            "nadir": SyntheticCamera("nadir", "CAMARA NADIR VIRTUAL", (30, 200, 255)),
            "sky": SyntheticCamera("sky", "CAMARA CENITAL VIRTUAL", (160, 120, 255)),
            "real": SyntheticCamera("real", "CAMARA REAL / ESP32-CAM", (80, 240, 140)),
        }

    def set_time(self, sim_time: float) -> None:
        self.nadir.set_time(sim_time)
        self.sky.set_time(sim_time)

    def get(self, name: str) -> CameraFeed:
        if name == "nadir":
            return self.nadir if self.nadir.latest_jpeg() else self._synthetic["nadir"]
        if name == "sky":
            return self.sky if self.sky.latest_jpeg() else self._synthetic["sky"]
        if name == "real":
            return self.real if self.real.latest_jpeg() else self._synthetic["real"]
        raise KeyError(name)

    def snapshot(self, name: str) -> Optional[bytes]:
        return self.get(name).latest_jpeg()

    def status(self) -> Dict[str, Any]:
        return {
            "nadir": self.nadir.status(),
            "sky": self.sky.status(),
            "real": self.real.status(),
        }


# ---------------------------------------------------------------------------
# Fuentes de datos
# ---------------------------------------------------------------------------
class BaseSource:
    name = "base"

    def __init__(self, hub: "RealtimeHub"):
        self.hub = hub

    def active(self) -> bool:
        """True solo mientras esta fuente siga siendo la seleccionada."""
        return self.hub.running and self.hub.source is self

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    async def run(self) -> None:
        raise NotImplementedError


class CessnaSimSource(BaseSource):
    """Reproduce el vuelo real registrado por JSBSim y el EKF."""

    name = "sim"

    def __init__(self, hub: "RealtimeHub"):
        super().__init__(hub)
        self.df_gt = pd.read_csv(DATA_DIR / "flight_log.csv")
        self.df_fused = pd.read_csv(DATA_DIR / "fused_navigation_log.csv")
        self.df_vis = pd.read_csv(DATA_DIR / "visual_navigation_results.csv")
        self.df_imu = pd.read_csv(DATA_DIR / "imu_measurements.csv")
        self.duration = float(self.df_gt["sim_time_sec"].max())
        self._gt_t = self.df_gt["sim_time_sec"].to_numpy(dtype=float)
        self._fused_t = self.df_fused["sim_time_sec"].to_numpy(dtype=float)
        self._imu_t = self.df_imu["sim_time_sec"].to_numpy(dtype=float)
        self._vis_t = self.df_vis["sim_time_sec"].to_numpy(dtype=float)
        self._center_lat = float(self.df_gt["lat_geod_deg"].iloc[0])
        self._center_lon = float(self.df_gt["lon_geod_deg"].iloc[0])
        # Origen métrico del EKF/VO: centro del mapa de referencia.
        with open(DATA_DIR / "reference_map_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._map_center_lat = float(meta["center_geodetic"]["latitude_deg"])
        self._map_center_lon = float(meta["center_geodetic"]["longitude_deg"])

    def frame_at(self, t: float) -> Dict[str, Any]:
        i = _nearest_index(self._gt_t, t)
        gt = self.df_gt.iloc[i]
        j = _nearest_index(self._fused_t, t)
        fused = self.df_fused.iloc[j]
        k = _nearest_index(self._imu_t, t)
        imu = self.df_imu.iloc[k]
        v = _nearest_index(self._vis_t, t)
        vis = self.df_vis.iloc[v]

        alt = float(gt["altitude_msl_m"])
        east = float(fused["gt_east_m"])
        north = float(fused["gt_north_m"])
        y = alt - BASE_ALT_CESSNA_M
        heading = float(gt["yaw_deg"]) % 360.0

        gt_pos = [east, y, -north]
        fused_pos = [float(fused["fused_east_m"]), -float(fused["fused_down_m"]) - BASE_ALT_CESSNA_M, -float(fused["fused_north_m"])]
        ins_pos = [float(fused["pure_ins_east_m"]), y, -float(fused["pure_ins_north_m"])]
        vo_pos = [float(vis["est_vo_east_m"]), y, -float(vis["est_vo_north_m"])]
        vo_fused_pos = [float(vis["est_fused_east_m"]), y, -float(vis["est_fused_north_m"])]

        return {
            "source": "sim",
            "mission": "cessna",
            "sim_time": round(t, 2),
            "vehicle": {
                "x": east,
                "y": y,
                "z": -north,
                "lat": float(gt["lat_geod_deg"]),
                "lon": float(gt["lon_geod_deg"]),
                "alt_m": alt,
                "agl_m": float(gt["altitude_agl_m"]),
                "roll": float(gt["roll_deg"]),
                "pitch": float(gt["pitch_deg"]),
                "yaw": heading,
                "heading": heading,
                "speed": float(gt["groundspeed_mps"]),
                "vn": float(gt["vel_north_mps"]),
                "ve": float(gt["vel_east_mps"]),
                "vd": float(gt["vel_down_mps"]),
            },
            "estimates": {
                "gt": {"x": gt_pos[0], "y": gt_pos[1], "z": gt_pos[2], "yaw": heading},
                "fused": {
                    "x": fused_pos[0],
                    "y": fused_pos[1],
                    "z": fused_pos[2],
                    "yaw": float(fused["fused_yaw_deg"]),
                    "sigma_m": math.hypot(float(fused["sigma_3_north_m"]), float(fused["sigma_3_east_m"])),
                },
                "ins": {"x": ins_pos[0], "y": ins_pos[1], "z": ins_pos[2], "yaw": heading},
                "vo": {"x": vo_pos[0], "y": vo_pos[1], "z": vo_pos[2], "yaw": float(vis["est_vo_yaw_deg"])},
                "vision": {"x": vo_fused_pos[0], "y": vo_fused_pos[1], "z": vo_fused_pos[2], "yaw": float(vis["est_fused_yaw_deg"])},
            },
            "imu": {
                "ax": float(imu["accel_x_mps2"]),
                "ay": float(imu["accel_y_mps2"]),
                "az": float(imu["accel_z_mps2"]),
                "gx": float(imu["gyro_p_rad_s"]),
                "gy": float(imu["gyro_q_rad_s"]),
                "gz": float(imu["gyro_r_rad_s"]),
            },
            "errors": {
                "fused_pos_m": float(fused["err_fused_pos_m"]),
                "ins_pos_m": float(fused["err_pure_ins_pos_m"]),
                "fused_yaw_deg": float(fused["err_fused_yaw_deg"]),
                "vo_pos_m": float(vis["error_vo_m"]),
            },
            "health": {
                "gps": True,
                "sats": 11,
                "rssi": -61,
                "battery": round(max(20.0, 100.0 - 0.18 * t), 1),
                "cpu": 24,
                "link": "nominal",
            },
            "camera_time": t,
        }

    async def run(self) -> None:
        self.hub.set_status(source=self.name, mission="cessna")
        t = 0.0
        while self.active():
            if not self.hub.paused:
                t += 0.05 * self.hub.speed
                if t > self.duration:
                    t = 0.0
                    self.hub.clear_trails()
                self.hub.publish(self.frame_at(t))
                self.hub.cameras.set_time(t)
            await asyncio.sleep(0.05)


class CampusSource(BaseSource):
    """Reproduce la misión de inspección del campus."""

    name = "campus"

    def __init__(self, hub: "RealtimeHub"):
        super().__init__(hub)
        if not MISSION_PATH.exists():
            from src.campus_scene import build_mission

            build_mission()
        with open(MISSION_PATH, "r", encoding="utf-8") as f:
            self.mission = json.load(f)
        self.samples: List[Dict[str, Any]] = self.mission["samples"]
        self.duration = float(self.mission["duration"])
        self._idx_t = [float(s["t"]) for s in self.samples]

    def frame_at(self, t: float) -> Dict[str, Any]:
        idx = _nearest_index(np.array(self._idx_t), t)
        s = self.samples[idx]
        return {
            "source": "campus",
            "mission": "campus",
            "sim_time": float(s["t"]),
            "vehicle": {
                "x": s["x"],
                "y": s["y"],
                "z": s["z"],
                "lat": s["lat"],
                "lon": s["lon"],
                "alt_m": s["agl"],
                "agl_m": s["agl"],
                "roll": s["roll"],
                "pitch": s["pitch"],
                "yaw": s["yaw"],
                "heading": s["yaw"],
                "speed": s["speed"],
                "vn": 0.0,
                "ve": 0.0,
                "vd": 0.0,
            },
            "estimates": {
                "gt": {"x": s["x"], "y": s["y"], "z": s["z"], "yaw": s["yaw"]},
                "fused": {"x": s["ekf"][0], "y": s["ekf"][1], "z": s["ekf"][2], "yaw": s["yaw"], "sigma_m": s["sigma_3_m"]},
                "ins": {"x": s["ins"][0], "y": s["ins"][1], "z": s["ins"][2], "yaw": s["yaw"]},
                "vo": {"x": s["vo"][0], "y": s["y"], "z": s["vo"][1], "yaw": s["yaw"]},
            },
            "imu": {"ax": 0.0, "ay": 0.0, "az": 0.0, "gx": 0.0, "gy": 0.0, "gz": 0.0},
            "errors": {
                "fused_pos_m": s["err_ekf_m"],
                "ins_pos_m": s["err_ins_m"],
                "fused_yaw_deg": 0.0,
                "vo_pos_m": 1.6,
            },
            "health": {
                "gps": s["gps"],
                "sats": s["sats"],
                "rssi": s["rssi"],
                "battery": s["battery"],
                "cpu": 38,
                "link": "nominal" if s["gps"] else "degradado",
            },
            "camera_time": float(s["t"]),
            "mission_meta": {
                "name": self.mission["name"],
                "duration": self.duration,
                "gps_denied": self.mission.get("gps_denied", []),
                "metrics": self.mission.get("metrics", {}),
            },
        }

    async def run(self) -> None:
        self.hub.set_status(source=self.name, mission="campus")
        t = 0.0
        while self.active():
            if not self.hub.paused:
                t += 0.05 * self.hub.speed
                if t > self.duration:
                    t = 0.0
                    self.hub.clear_trails()
                self.hub.publish(self.frame_at(t))
                # La cámara virtual sigue el tiempo de la misión (mod 150 s).
                self.hub.cameras.set_time(t % 150.0)
            await asyncio.sleep(0.05)


class SerialSource(BaseSource):
    """IMU física por puerto serie; integra actitud y posición en vivo."""

    name = "serial"

    def __init__(self, hub: "RealtimeHub", port: str = "auto", baud: int = 115200):
        super().__init__(hub)
        self.port = port
        self.baud = baud
        self.dr = DeadReckoning()
        self._reader = None
        self._thread: Optional[threading.Thread] = None

    def _run_reader(self) -> None:
        if ESP32IMUReader is None:
            print("[serial] pyserial no disponible; use --source live")
            return
        port = self.port
        if port in ("auto", "", None):
            ports = list_available_ports()
            port = ports[0][0] if ports else "auto"
        reader = ESP32IMUReader(port=port, baudrate=self.baud)
        connected = reader.connect()
        if not connected:
            # Modo demostración: emula el ESP32 para validar todo el flujo.
            print("[serial] Sin hardware; usando emulador ESP32 (50 Hz). Conecte la IMU para datos reales.")
            try:
                from src.serial_imu_reader import simulate_esp32_stream
            except Exception:
                try:
                    from serial_imu_reader import simulate_esp32_stream  # type: ignore
                except Exception:
                    return
            for packet in simulate_esp32_stream(duration_sec=3600.0):
                if not self.active():
                    break
                self._handle_packet(packet)
            return
        self._reader = reader
        print(f"[serial] Leyendo IMU en {reader.port} a {self.baud} baudios...")
        while self.active():
            packet = reader.read_packet()
            if packet:
                self._handle_packet(packet)
        try:
            reader.disconnect()
        except Exception:
            pass

    def _handle_packet(self, packet: Dict[str, Any]) -> None:
        accel = (packet["accel_x_mps2"], packet["accel_y_mps2"], packet["accel_z_mps2"])
        gyro = (packet["gyro_p_rad_s"], packet["gyro_q_rad_s"], packet["gyro_r_rad_s"])
        attitude = None
        if all(k in packet for k in ("roll_deg", "pitch_deg", "yaw_deg")):
            attitude = (packet["roll_deg"], packet["pitch_deg"], packet["yaw_deg"])
        self.dr.update(accel, gyro, attitude)
        frame = self.hub.frame_from_dead_reckoning(self.dr, "serial", packet)
        self.hub.publish(frame)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_reader, daemon=True)
        self._thread.start()

    async def run(self) -> None:
        self.hub.set_status(source=self.name, mission="hardware")
        while self.active():
            await asyncio.sleep(0.2)

    def stop(self) -> None:
        if self._reader is not None:
            try:
                self._reader.disconnect()
            except Exception:
                pass


class LiveSource(BaseSource):
    """Espera telemetría externa por HTTP/UDP y la publica."""

    name = "live"

    def __init__(self, hub: "RealtimeHub"):
        super().__init__(hub)

    async def run(self) -> None:
        self.hub.set_status(source=self.name, mission="hardware")
        while self.active():
            await asyncio.sleep(0.25)


class ReplaySource(BaseSource):
    """Reproduce una sesión grabada (telemetry.csv)."""

    name = "replay"

    def __init__(self, hub: "RealtimeHub", session: Optional[str] = None):
        super().__init__(hub)
        self.session = session
        self.rows: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not SESSIONS_DIR.exists():
            return
        sessions = sorted([p for p in SESSIONS_DIR.iterdir() if p.is_dir()], reverse=True)
        if not sessions:
            return
        path = SESSIONS_DIR / self.session if self.session else sessions[0]
        csv_path = path / "telemetry.csv"
        if not csv_path.exists():
            return
        with open(csv_path, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        print(f"[replay] {len(self.rows)} muestras de {csv_path}")

    async def run(self) -> None:
        self.hub.set_status(source=self.name, mission="hardware")
        if not self.rows:
            while self.active():
                await asyncio.sleep(0.5)
            return
        i = 0
        while self.active():
            if not self.hub.paused:
                row = self.rows[i % len(self.rows)]
                payload = {}
                for key, value in row.items():
                    try:
                        payload[key] = float(value)
                    except (TypeError, ValueError):
                        payload[key] = value
                self.hub.ingest_telemetry(payload, source="replay", broadcast=True)
                i += 1
            await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# Estado y grabación
# ---------------------------------------------------------------------------
class TelemetryState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.latest: Optional[Dict[str, Any]] = None
        self.history: deque = deque(maxlen=4000)
        self.seq = 0
        self.source = "sim"
        self.mission = "cessna"
        self.paused = False
        self.speed = 1.0
        self.running = True
        self.recording = False
        self.record_dir: Optional[Path] = None
        self.record_writer = None
        self.record_file = None
        self.record_count = 0
        self.clear_trails_flag = False
        self.started_at = utc_now_iso()

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "latest": self.latest,
                "seq": self.seq,
                "source": self.source,
                "mission": self.mission,
                "paused": self.paused,
                "speed": self.speed,
                "recording": self.recording,
                "record_dir": str(self.record_dir) if self.record_dir else None,
                "record_count": self.record_count,
                "clear_trails": self.clear_trails_flag,
            }


# ---------------------------------------------------------------------------
# Hub
# ---------------------------------------------------------------------------
class RealtimeHub:
    def __init__(
        self,
        host: str = "127.0.0.1",
        http_port: int = 8080,
        ws_port: int = 8081,
        source: str = "sim",
        mission: str = "cessna",
        serial_port: str = "auto",
        baud: int = 115200,
        udp_port: int = 9876,
        record_frames: bool = False,
    ):
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port
        self.source_name = source
        self.mission = mission
        self.serial_port = serial_port
        self.baud = baud
        self.udp_port = udp_port
        self.record_frames = record_frames
        self.state = TelemetryState()
        self.cameras = CameraManager()
        self.clients: set = set()
        self.source: Optional[BaseSource] = None
        self.http_server: Optional[ThreadingHTTPServer] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_server = None
        self._udp_transport = None
        self.scene: Optional[Dict[str, Any]] = None
        self.config = self._build_config()

    # -- estado ------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.state.running

    @property
    def paused(self) -> bool:
        return self.state.paused

    @property
    def speed(self) -> float:
        return self.state.speed

    def set_status(self, source: str, mission: str) -> None:
        with self.state.lock:
            self.state.source = source
            self.state.mission = mission
        print(f"[hub] fuente={source} misión={mission}")

    def clear_trails(self) -> None:
        with self.state.lock:
            self.state.clear_trails_flag = True

    def _build_config(self) -> Dict[str, Any]:
        return {
            "http_port": self.http_port,
            "ws_port": self.ws_port,
            "udp_port": self.udp_port,
            "missions": [
                {"id": "cessna", "name": "Cessna 172P · Circuito JSBSim", "kind": "aircraft"},
                {"id": "campus", "name": "Dron · Inspección TecMM Zapopan", "kind": "drone"},
                {"id": "hardware", "name": "Hardware en vivo · ESP32/STM32", "kind": "hardware"},
            ],
            "sources": [
                {"id": "sim", "name": "Simulación Cessna (datos reales)"},
                {"id": "campus", "name": "Misión campus (simulación)"},
                {"id": "serial", "name": "IMU física por puerto serie"},
                {"id": "live", "name": "Telemetría externa HTTP/UDP"},
                {"id": "replay", "name": "Reproducir sesión grabada"},
            ],
            "cameras": ["nadir", "sky", "real"],
            "camera_status": self.cameras.status(),
            "serial_ports": list_available_ports(),
            "started_at": self.state.started_at,
            "protocol": {
                "websocket": "state a 20 Hz",
                "http_telemetry": "POST /api/telemetry",
                "udp": f"JSON por UDP al puerto {self.udp_port}",
            },
        }

    def publish(self, frame: Dict[str, Any]) -> None:
        with self.state.lock:
            self.state.seq += 1
            frame["seq"] = self.state.seq
            frame["ts"] = round(time.time(), 3)
            frame["source"] = frame.get("source", self.state.source)
            frame["mission"] = frame.get("mission", self.state.mission)
            self.state.latest = frame
            self.state.history.append(frame)
            if self.state.recording:
                self._record_frame(frame)

    # -- grabación ---------------------------------------------------------
    def start_recording(self) -> Path:
        with self.state.lock:
            if self.state.recording and self.state.record_dir:
                return self.state.record_dir
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = SESSIONS_DIR / f"session_{stamp}"
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / "frames").mkdir(exist_ok=True)
            csv_path = session_dir / "telemetry.csv"
            fields = [
                "ts", "seq", "source", "mission", "sim_time",
                "x", "y", "z", "lat", "lon", "alt_m", "agl_m",
                "roll", "pitch", "yaw", "speed",
                "ax", "ay", "az", "gx", "gy", "gz",
                "err_fused_m", "err_ins_m", "gps", "sats", "rssi", "battery",
            ]
            self.state.record_file = open(csv_path, "w", newline="", encoding="utf-8")
            self.state.record_writer = csv.DictWriter(self.state.record_file, fieldnames=fields)
            self.state.record_writer.writeheader()
            self.state.record_dir = session_dir
            self.state.recording = True
            self.state.record_count = 0
            with open(session_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "started_at": utc_now_iso(),
                        "source": self.state.source,
                        "mission": self.state.mission,
                        "config": self.config,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"[record] Grabando en {session_dir}")
            return session_dir

    def stop_recording(self) -> Optional[str]:
        with self.state.lock:
            if not self.state.recording:
                return None
            self.state.recording = False
            if self.state.record_file:
                self.state.record_file.flush()
                self.state.record_file.close()
            self.state.record_file = None
            self.state.record_writer = None
            path = str(self.state.record_dir) if self.state.record_dir else None
            print(f"[record] Sesión guardada: {path}")
            return path

    def _record_frame(self, frame: Dict[str, Any]) -> None:
        writer = self.state.record_writer
        if writer is None:
            return
        vehicle = frame.get("vehicle", {})
        imu = frame.get("imu", {})
        errors = frame.get("errors", {})
        health = frame.get("health", {})
        writer.writerow(
            {
                "ts": frame.get("ts"),
                "seq": frame.get("seq"),
                "source": frame.get("source"),
                "mission": frame.get("mission"),
                "sim_time": frame.get("sim_time"),
                "x": vehicle.get("x"),
                "y": vehicle.get("y"),
                "z": vehicle.get("z"),
                "lat": vehicle.get("lat"),
                "lon": vehicle.get("lon"),
                "alt_m": vehicle.get("alt_m"),
                "agl_m": vehicle.get("agl_m"),
                "roll": vehicle.get("roll"),
                "pitch": vehicle.get("pitch"),
                "yaw": vehicle.get("yaw"),
                "speed": vehicle.get("speed"),
                "ax": imu.get("ax"),
                "ay": imu.get("ay"),
                "az": imu.get("az"),
                "gx": imu.get("gx"),
                "gy": imu.get("gy"),
                "gz": imu.get("gz"),
                "err_fused_m": errors.get("fused_pos_m"),
                "err_ins_m": errors.get("ins_pos_m"),
                "gps": health.get("gps"),
                "sats": health.get("sats"),
                "rssi": health.get("rssi"),
                "battery": health.get("battery"),
            }
        )
        self.state.record_count += 1
        # Guardar un fotograma cada 20 muestras (~1 Hz) para evidencia visual.
        if self.record_frames and self.state.record_count % 20 == 0:
            for cam in ("nadir", "real"):
                jpeg = self.cameras.snapshot(cam)
                if jpeg:
                    path = self.state.record_dir / "frames" / f"{cam}_{self.state.record_count:06d}.jpg"
                    path.write_bytes(jpeg)

    # -- ingesta -----------------------------------------------------------
    def frame_from_dead_reckoning(
        self, dr: DeadReckoning, source: str, packet: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        packet = packet or {}
        lat = packet.get("latitude_deg")
        lon = packet.get("longitude_deg")
        x, z = dr.x, dr.z
        if lat is not None and lon is not None:
            # Si el firmware entrega GPS, se ancla al campus.
            try:
                x, z = project_geodetic(float(lat), float(lon), 20.703454, -103.474731, 110713.0, 104218.0)
            except Exception:
                pass
        y = max(0.0, dr.y)
        yaw = packet.get("yaw_deg", dr.yaw) % 360.0
        speed = math.sqrt(dr.vx ** 2 + dr.vy ** 2 + dr.vz ** 2)
        return {
            "source": source,
            "mission": "hardware",
            "sim_time": round(time.monotonic() % 100000, 2),
            "vehicle": {
                "x": round(x, 2),
                "y": round(y, 2),
                "z": round(z, 2),
                "lat": float(lat) if lat is not None else None,
                "lon": float(lon) if lon is not None else None,
                "alt_m": round(y, 2),
                "agl_m": round(y, 2),
                "roll": round(dr.roll, 2),
                "pitch": round(dr.pitch, 2),
                "yaw": round(yaw, 2),
                "heading": round(yaw, 2),
                "speed": round(speed, 2),
                "vn": round(-dr.vz, 2),
                "ve": round(dr.vx, 2),
                "vd": round(-dr.vy, 2),
            },
            "estimates": {
                "gt": {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2), "yaw": round(yaw, 2)},
                "fused": {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2), "yaw": round(yaw, 2), "sigma_m": 3.0},
                "ins": {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2), "yaw": round(yaw, 2)},
            },
            "imu": {
                "ax": float(packet.get("accel_x_mps2", 0.0)),
                "ay": float(packet.get("accel_y_mps2", 0.0)),
                "az": float(packet.get("accel_z_mps2", 0.0)),
                "gx": float(packet.get("gyro_p_rad_s", 0.0)),
                "gy": float(packet.get("gyro_q_rad_s", 0.0)),
                "gz": float(packet.get("gyro_r_rad_s", 0.0)),
            },
            "errors": {"fused_pos_m": 0.0, "ins_pos_m": 0.0, "fused_yaw_deg": 0.0, "vo_pos_m": 0.0},
            "health": {
                "gps": lat is not None,
                "sats": 9 if lat is not None else 0,
                "rssi": -60,
                "battery": 88.0,
                "cpu": 35,
                "link": "serial",
            },
            "camera_time": round(time.monotonic() % 150.0, 2),
        }

    def ingest_telemetry(self, payload: Dict[str, Any], source: str = "live", broadcast: bool = True) -> Dict[str, Any]:
        """Acepta un diccionario de telemetría (IMU/GPS/actitud) y lo publica."""
        # Aplanar estructuras anidadas simples.
        flat: Dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{key}_{sub_key}"] = sub_value
            else:
                flat[key] = value

        def pick(*names: str, default: Any = None) -> Any:
            for name in names:
                if name in flat and flat[name] not in (None, ""):
                    try:
                        return float(flat[name])
                    except (TypeError, ValueError):
                        return flat[name]
            return default

        lat = pick("lat", "latitude_deg", "lat_geod_deg")
        lon = pick("lon", "longitude_deg", "lon_geod_deg")
        x = pick("x", "east_m", "pos_east")
        y = pick("y", "up_m", "alt_agl_m", "altitude_agl_m")
        z = pick("z", "south_m")
        roll = pick("roll", "roll_deg", default=0.0)
        pitch = pick("pitch", "pitch_deg", default=0.0)
        yaw = pick("yaw", "heading", "yaw_deg", default=0.0)
        ax = pick("ax", "accel_x_mps2", default=0.0)
        ay = pick("ay", "accel_y_mps2", default=0.0)
        az = pick("az", "accel_z_mps2", default=0.0)
        gx = pick("gx", "gyro_p_rad_s", default=0.0)
        gy = pick("gy", "gyro_q_rad_s", default=0.0)
        gz = pick("gz", "gyro_r_rad_s", default=0.0)

        # Integrador compartido para paquetes sin posición.
        if not hasattr(self, "_live_dr"):
            self._live_dr = DeadReckoning()
        dr = self._live_dr
        if x is None or y is None or z is None:
            dr.update((ax, ay, az), (gx, gy, gz), (roll, pitch, yaw))
            x, y, z = dr.x, dr.y, dr.z
        else:
            dr.x, dr.y, dr.z = float(x), float(y), float(z)

        if lat is not None and lon is not None:
            px, pz = project_geodetic(float(lat), float(lon), 20.703454, -103.474731, 110713.0, 104218.0)
            x, z = px, pz

        sim_time = pick("sim_time", "t", default=round(time.monotonic() % 100000, 2))
        frame = {
            "source": source,
            "mission": "hardware",
            "sim_time": float(sim_time),
            "vehicle": {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "lat": float(lat) if lat is not None else None,
                "lon": float(lon) if lon is not None else None,
                "alt_m": float(y),
                "agl_m": float(y),
                "roll": float(roll),
                "pitch": float(pitch),
                "yaw": float(yaw) % 360.0,
                "heading": float(yaw) % 360.0,
                "speed": float(math.sqrt(dr.vx ** 2 + dr.vy ** 2 + dr.vz ** 2)),
                "vn": float(-dr.vz),
                "ve": float(dr.vx),
                "vd": float(-dr.vy),
            },
            "estimates": {
                "gt": {"x": float(x), "y": float(y), "z": float(z), "yaw": float(yaw) % 360.0},
                "fused": {"x": float(x), "y": float(y), "z": float(z), "yaw": float(yaw) % 360.0, "sigma_m": 2.0},
                "ins": {"x": float(x), "y": float(y), "z": float(z), "yaw": float(yaw) % 360.0},
            },
            "imu": {"ax": float(ax), "ay": float(ay), "az": float(az), "gx": float(gx), "gy": float(gy), "gz": float(gz)},
            "errors": {
                "fused_pos_m": float(pick("err_fused_m", "error_m", default=0.0)),
                "ins_pos_m": float(pick("err_ins_m", default=0.0)),
                "fused_yaw_deg": 0.0,
                "vo_pos_m": 0.0,
            },
            "health": {
                "gps": bool(pick("gps", default=lat is not None)),
                "sats": int(pick("sats", default=9)),
                "rssi": float(pick("rssi", default=-60)),
                "battery": float(pick("battery", default=90)),
                "cpu": float(pick("cpu", default=30)),
                "link": str(flat.get("link", "live")),
            },
            "camera_time": float(sim_time),
            "raw": {k: v for k, v in flat.items() if isinstance(v, (int, float, str, bool))},
        }
        if broadcast:
            self.publish(frame)
        return frame

    # -- fuentes -----------------------------------------------------------
    def _make_source(self, name: str) -> BaseSource:
        if name == "campus":
            return CampusSource(self)
        if name == "serial":
            return SerialSource(self, self.serial_port, self.baud)
        if name == "live":
            return LiveSource(self)
        if name == "replay":
            return ReplaySource(self)
        return CessnaSimSource(self)

    def switch_source(self, name: str, mission: Optional[str] = None) -> None:
        if name == self.source_name and self.source is not None:
            return
        print(f"[hub] Cambiando fuente a {name}")
        if self.source is not None:
            try:
                self.source.stop()
            except Exception:
                pass
        self.source_name = name
        if mission:
            self.mission = mission
        self.state.running = True
        self.source = self._make_source(name)
        self.source.start()
        if self._loop is not None:
            self._loop.create_task(self.source.run())

    # -- UDP ---------------------------------------------------------------
    async def _udp_server(self) -> None:
        class Protocol(asyncio.DatagramProtocol):
            def __init__(self, hub: "RealtimeHub"):
                self.hub = hub

            def datagram_received(self, data: bytes, addr) -> None:
                try:
                    payload = json.loads(data.decode("utf-8", errors="ignore"))
                    if isinstance(payload, dict):
                        self.hub.ingest_telemetry(payload, source="live", broadcast=True)
                except Exception:
                    pass

        try:
            self._udp_transport, _ = await self._loop.create_datagram_endpoint(
                lambda: Protocol(self), local_addr=(self.host, self.udp_port)
            )
            print(f"[hub] UDP escuchando en {self.host}:{self.udp_port}")
        except Exception as exc:
            print(f"[hub] UDP no disponible en {self.udp_port}: {exc}")

    # -- WebSocket ---------------------------------------------------------
    async def _ws_handler(self, websocket) -> None:
        self.clients.add(websocket)
        try:
            with self.state.lock:
                latest = self.state.latest
                config = dict(self.config)
                config["camera_status"] = self.cameras.status()
            await websocket.send(json.dumps({"type": "hello", "config": config}, default=json_default))
            if latest:
                await websocket.send(json.dumps({"type": "state", "data": latest}, default=json_default))
            async for message in websocket:
                try:
                    data = json.loads(message)
                except Exception:
                    continue
                if data.get("type") == "ping":
                    await websocket.send(json.dumps({"type": "pong", "ts": time.time()}))
                elif data.get("type") == "get_state":
                    with self.state.lock:
                        latest = self.state.latest
                    if latest:
                        await websocket.send(json.dumps({"type": "state", "data": latest}, default=json_default))
        except Exception:
            pass
        finally:
            self.clients.discard(websocket)

    async def _broadcast_loop(self) -> None:
        last_seq = -1
        while self.running:
            if self.clients:
                with self.state.lock:
                    latest = self.state.latest
                    seq = self.state.seq
                    clear = self.state.clear_trails_flag
                    if clear:
                        self.state.clear_trails_flag = False
                if latest is not None and (seq != last_seq or clear):
                    payload = json.dumps(
                        {"type": "state", "data": latest, "clear_trails": clear},
                        default=json_default,
                    )
                    await asyncio.gather(
                        *[self._safe_send(client, payload) for client in list(self.clients)],
                        return_exceptions=True,
                    )
                    last_seq = seq
            await asyncio.sleep(0.04)

    @staticmethod
    async def _safe_send(client, payload: str) -> None:
        try:
            await client.send(payload)
        except Exception:
            pass

    # -- HTTP --------------------------------------------------------------
    def _start_http(self) -> None:
        hub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"
            server_version = "TecMMHub/2.0"

            def log_message(self, fmt, *args):  # noqa: N802
                pass

            def _send(self, code: int, body: bytes, content_type: str, extra: Optional[Dict[str, str]] = None):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store")
                if extra:
                    for key, value in extra.items():
                        self.send_header(key, value)
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj: Any, code: int = 200):
                self._send(code, json.dumps(obj, default=json_default, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")

            def _serve_file(self, path: Path):
                if not path.exists() or not path.is_file():
                    self._json({"error": "not found", "path": str(path)}, 404)
                    return
                content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                try:
                    body = path.read_bytes()
                except Exception as exc:
                    self._json({"error": str(exc)}, 500)
                    return
                self._send(200, body, content_type)

            def _safe_static(self, root: Path, relative: str) -> Optional[Path]:
                try:
                    candidate = (root / relative.lstrip("/\\")).resolve()
                    candidate.relative_to(root.resolve())
                    return candidate
                except Exception:
                    return None

            def do_OPTIONS(self):  # noqa: N802
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_GET(self):  # noqa: N802
                parsed = urlparse(self.path)
                path = parsed.path
                if path in ("/", "/index.html", "/dashboard"):
                    self._serve_file(WEB_DIR / "dashboard.html")
                    return
                if path == "/favicon.ico":
                    self._send(204, b"", "image/x-icon")
                    return
                if path.startswith("/api/"):
                    self._handle_api_get(path, parse_qs(parsed.query))
                    return
                if path.startswith("/camera/"):
                    self._handle_camera(path)
                    return
                if path.startswith("/web/"):
                    candidate = self._safe_static(WEB_DIR, path[len("/web/"):])
                    if candidate:
                        self._serve_file(candidate)
                    else:
                        self._json({"error": "forbidden"}, 403)
                    return
                if path.startswith("/data/"):
                    candidate = self._safe_static(DATA_DIR, path[len("/data/"):])
                    if candidate:
                        self._serve_file(candidate)
                    else:
                        self._json({"error": "forbidden"}, 403)
                    return
                self._json({"error": "not found"}, 404)

            def do_POST(self):  # noqa: N802
                parsed = urlparse(self.path)
                length = int(self.headers.get("Content-Length", "0") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    payload = json.loads(raw.decode("utf-8")) if raw else {}
                except Exception:
                    payload = {}
                if parsed.path == "/api/telemetry":
                    frame = hub.ingest_telemetry(payload, source="live", broadcast=True)
                    self._json({"ok": True, "seq": hub.state.seq, "frame": frame})
                    return
                if parsed.path == "/api/control":
                    self._json(hub.handle_control(payload))
                    return
                self._json({"error": "not found"}, 404)

            def _handle_api_get(self, path: str, query: Dict[str, List[str]]):
                if path == "/api/status":
                    with hub.state.lock:
                        snapshot = hub.state.snapshot()
                    snapshot["clients"] = len(hub.clients)
                    snapshot["cameras"] = hub.cameras.status()
                    snapshot["config"] = hub.config
                    snapshot["serial_ports"] = list_available_ports()
                    self._json(snapshot)
                    return
                if path == "/api/config":
                    self._json(hub.config)
                    return
                if path == "/api/scene":
                    mission = (query.get("mission") or ["campus"])[0]
                    if mission == "cessna":
                        self._json(
                            {
                                "kind": "cessna",
                                "map": "/data/reference_map.png",
                                "width_m": 8000.0,
                                "height_m": 8000.0,
                                "center": [20.7768, -103.3365],
                                "runway": {"lat": 20.778, "lon": -103.333, "heading_deg": 20.0},
                            }
                        )
                    else:
                        if hub.scene is None:
                            if SCENE_PATH.exists():
                                with open(SCENE_PATH, "r", encoding="utf-8") as f:
                                    hub.scene = json.load(f)
                            else:
                                from src.campus_scene import ensure_scene

                                hub.scene = ensure_scene()
                        self._json(hub.scene)
                    return
                if path == "/api/mission":
                    name = (query.get("name") or ["campus"])[0]
                    if name == "campus" and MISSION_PATH.exists():
                        with open(MISSION_PATH, "r", encoding="utf-8") as f:
                            mission = json.load(f)
                        # No enviar las 1600 muestras: el WebSocket ya las empuja.
                        mission.pop("samples", None)
                        self._json(mission)
                    else:
                        self._json({"error": "misión no disponible"}, 404)
                    return
                if path == "/api/telemetry/latest":
                    with hub.state.lock:
                        self._json(hub.state.latest or {})
                    return
                if path == "/api/telemetry/history":
                    limit = int((query.get("limit") or ["300"])[0])
                    with hub.state.lock:
                        history = list(hub.state.history)[-limit:]
                    self._json({"count": len(history), "samples": history})
                    return
                if path == "/api/sessions":
                    sessions = []
                    if SESSIONS_DIR.exists():
                        for item in sorted(SESSIONS_DIR.iterdir(), reverse=True):
                            if item.is_dir():
                                meta = item / "meta.json"
                                info = {"name": item.name, "path": str(item)}
                                if meta.exists():
                                    try:
                                        with open(meta, "r", encoding="utf-8") as f:
                                            info["meta"] = json.load(f)
                                    except Exception:
                                        pass
                                sessions.append(info)
                    self._json({"sessions": sessions})
                    return
                if path == "/api/health":
                    self._json({"ok": True, "ts": time.time(), "clients": len(hub.clients)})
                    return
                self._json({"error": "not found"}, 404)

            def _handle_camera(self, path: str):
                name = path[len("/camera/"):].split(".")[0]
                if name not in ("nadir", "sky", "real"):
                    self._json({"error": "cámara desconocida"}, 404)
                    return
                if path.endswith(".jpg"):
                    jpeg = hub.cameras.snapshot(name)
                    if jpeg is None:
                        self._json({"error": "sin señal"}, 503)
                    else:
                        self._send(200, jpeg, "image/jpeg")
                    return
                if not path.endswith(".mjpg"):
                    self._json({"error": "formato no soportado"}, 404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Pragma", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try:
                    last = None
                    while hub.running:
                        jpeg = hub.cameras.snapshot(name)
                        if jpeg is None:
                            time.sleep(0.2)
                            continue
                        if jpeg is last:
                            time.sleep(0.08)
                            continue
                        last = jpeg
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                        time.sleep(0.08)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.http_server = Server((self.host, self.http_port), Handler)
        thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
        thread.start()
        print(f"[hub] Panel: http://{self.host if self.host != '0.0.0.0' else 'localhost'}:{self.http_port}")

    # -- control -----------------------------------------------------------
    def handle_control(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        action = payload.get("action")
        if action == "set_source":
            source = payload.get("source", "sim")
            mission = payload.get("mission")
            self.switch_source(source, mission)
            return {"ok": True, "source": source, "mission": mission}
        if action == "set_mission":
            mission = payload.get("mission", "campus")
            source = "campus" if mission == "campus" else ("sim" if mission == "cessna" else "live")
            self.switch_source(source, mission)
            return {"ok": True, "mission": mission, "source": source}
        if action == "play":
            self.state.paused = False
            return {"ok": True, "paused": False}
        if action == "pause":
            self.state.paused = True
            return {"ok": True, "paused": True}
        if action == "set_speed":
            self.state.speed = max(0.1, min(8.0, float(payload.get("speed", 1.0))))
            return {"ok": True, "speed": self.state.speed}
        if action == "reset":
            if hasattr(self, "_live_dr"):
                self._live_dr.reset()
            self.clear_trails()
            return {"ok": True}
        if action == "record_start":
            path = self.start_recording()
            return {"ok": True, "path": str(path)}
        if action == "record_stop":
            path = self.stop_recording()
            return {"ok": True, "path": path}
        if action == "camera_config":
            name = payload.get("name", "real")
            if name != "real":
                return {"ok": False, "error": "solo la cámara real es configurable"}
            source = payload.get("source")
            if source is None:
                return {"ok": False, "error": "falta source"}
            if isinstance(source, str) and source.isdigit():
                source = int(source)
            self.cameras.real.configure(source)
            return {"ok": True, "camera": self.cameras.real.status()}
        if action == "calibrate":
            # Reset del integrador y de la actitud del filtro.
            if hasattr(self, "_live_dr"):
                self._live_dr.reset()
            return {"ok": True}
        return {"ok": False, "error": f"acción desconocida: {action}"}

    # -- arranque ----------------------------------------------------------
    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._start_http()
        if websockets is not None:
            self._ws_server = await websockets.serve(
                self._ws_handler, self.host, self.ws_port, max_size=4 * 1024 * 1024
            )
            print(f"[hub] WebSocket: ws://{self.host if self.host != '0.0.0.0' else 'localhost'}:{self.ws_port}")
        else:
            print("[hub] ADVERTENCIA: websockets no instalado; el panel usará polling REST.")
        await self._udp_server()
        self.source = self._make_source(self.source_name)
        self.source.start()
        tasks = [asyncio.create_task(self.source.run()), asyncio.create_task(self._broadcast_loop())]
        print("[hub] Listo. Ctrl+C para detener.")
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.state.running = False
        self.stop_recording()
        if self.source is not None:
            try:
                self.source.stop()
            except Exception:
                pass
        self.cameras.real.stop()
        if self.http_server is not None:
            self.http_server.shutdown()
        print("[hub] Detenido.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hub en tiempo real de navegación aérea")
    parser.add_argument("--host", default="127.0.0.1", help="Interfaz de escucha (0.0.0.0 para LAN)")
    parser.add_argument("--http-port", type=int, default=8080)
    parser.add_argument("--ws-port", type=int, default=8081)
    parser.add_argument("--udp-port", type=int, default=9876)
    parser.add_argument("--source", choices=["sim", "campus", "serial", "live", "replay"], default="sim")
    parser.add_argument("--mission", choices=["cessna", "campus", "hardware"], default=None)
    parser.add_argument("--port", default="auto", help="Puerto serie para --source serial (COM3, /dev/ttyUSB0, auto)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--camera", default=None, help="Webcam (índice) o URL MJPEG para la cámara real")
    parser.add_argument("--record-frames", action="store_true", help="Guardar fotogramas de evidencia ~1 Hz")
    args = parser.parse_args()

    source = args.source
    mission = args.mission or ("campus" if source == "campus" else ("hardware" if source in ("serial", "live") else "cessna"))
    hub = RealtimeHub(
        host=args.host,
        http_port=args.http_port,
        ws_port=args.ws_port,
        source=source,
        mission=mission,
        serial_port=args.port,
        baud=args.baud,
        udp_port=args.udp_port,
        record_frames=args.record_frames,
    )
    if args.camera is not None:
        camera_source: Any = int(args.camera) if str(args.camera).isdigit() else args.camera
        hub.cameras.real.configure(camera_source)

    try:
        asyncio.run(hub.run())
    except KeyboardInterrupt:
        hub.shutdown()


if __name__ == "__main__":
    main()
