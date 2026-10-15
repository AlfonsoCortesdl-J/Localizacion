"""
camera_models.py
================
Modelos matemáticos de cámaras virtuales para navegación aérea:
1. VirtualGroundCamera (Nadir): Cámara orientada al terreno para navegación relativa
   a imágenes (Terrain-Relative Navigation / Visual Odometry).
2. VirtualSkyCamera (Zenith / Star Tracker): Cámara orientada a la bóveda celeste
   para determinación de actitud (Star Tracker / Navegación Astronómica).

Modelado físico:
----------------
- Modelo Pinhole ideal con matriz intrínseca K.
- Proyección proyectiva de homografía plana H_map_to_cam para el terreno.
- Proyección de catálogo estelar mediante rotación ECI -> NED -> Body -> Camera
  con funciones de dispersión de punto (PSF Gaussianas) y ruido de sensor.
"""

import os
import json
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple, List


class PinholeCamera:
    """Modelo de cámara estenopeica (Pinhole) con matriz intrínseca K."""

    def __init__(self, width: int = 640, height: int = 480, fov_deg: float = 70.0):
        self.width = width
        self.height = height
        self.fov_deg = fov_deg

        # Longitud focal en píxeles a partir del campo de visión horizontal
        self.fx = (self.width / 2.0) / np.tan(np.radians(self.fov_deg / 2.0))
        self.fy = self.fx  # Píxeles cuadrados
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

        # Matriz intrínseca K
        self.K = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.K_inv = np.linalg.inv(self.K)


class VirtualGroundCamera(PinholeCamera):
    """
    Cámara virtual orientada al terreno (Nadir).
    Proyecta un ortomapa georreferenciado sobre el sensor de la cámara
    considerando la pose 6-DOF de la aeronave (Lat, Lon, Alt AGL, Roll, Pitch, Yaw).
    """

    METERS_PER_DEG_LAT = 111320.0

    def __init__(
        self,
        reference_map_path: str,
        meta_json_path: str,
        width: int = 640,
        height: int = 480,
        fov_deg: float = 70.0
    ):
        super().__init__(width, height, fov_deg)

        if not os.path.exists(reference_map_path):
            raise FileNotFoundError(f"Mapa de referencia no encontrado: {reference_map_path}")
        if not os.path.exists(meta_json_path):
            raise FileNotFoundError(f"Metadatos del mapa no encontrados: {meta_json_path}")

        self.map_img = cv2.imread(reference_map_path)
        with open(meta_json_path, "r", encoding="utf-8") as f:
            self.map_meta = json.load(f)

        self.map_w_px = self.map_meta["width_px"]
        self.map_h_px = self.map_meta["height_px"]
        self.gsd = self.map_meta["gsd_m_per_px"]
        self.center_lat = self.map_meta["center_geodetic"]["latitude_deg"]
        self.center_lon = self.map_meta["center_geodetic"]["longitude_deg"]
        self.meters_per_deg_lon = self.METERS_PER_DEG_LAT * np.cos(np.radians(self.center_lat))

        # Matriz de montaje Nadir en ejes cuerpo (Body -> Camera Optical Frame):
        # En FRD (X forward, Y right, Z down):
        # Eje óptico Z_c mira hacia abajo (+Z_b)
        # Eje horizontal X_c apunta hacia el ala derecha (+Y_b)
        # Eje vertical Y_c apunta hacia la cola (-X_b) para que el frente del avión quede arriba en la imagen
        self.R_b_c = np.array([
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def render_frame(
        self,
        lat_deg: float,
        lon_deg: float,
        altitude_agl_m: float,
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
        add_sensor_noise: bool = True
    ) -> np.ndarray:
        """
        Renderiza el fotograma de terreno mediante homografía proyectiva exacta.
        """
        # 1. Desplazamiento métrico respecto al centro del mapa
        dx_m = (lon_deg - self.center_lon) * self.meters_per_deg_lon  # Este (+)
        dy_m = (lat_deg - self.center_lat) * self.METERS_PER_DEG_LAT  # Norte (+)

        # 2. Rotación de actitud aeroespacial NED -> Body Frame (Z-Y-X)
        phi = np.radians(roll_deg)
        theta = np.radians(pitch_deg)
        psi = np.radians(yaw_deg)

        R_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(phi), -np.sin(phi)],
            [0.0, np.sin(phi), np.cos(phi)]
        ])
        R_y = np.array([
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)]
        ])
        R_z = np.array([
            [np.cos(psi), -np.sin(psi), 0.0],
            [np.sin(psi), np.cos(psi), 0.0],
            [0.0, 0.0, 1.0]
        ])
        R_ned_b = R_x @ R_y @ R_z

        # Rotación completa NED -> Camera
        R_ned_c = self.R_b_c @ R_ned_b

        # 3. Matriz de proyección afín desde píxeles del mapa a coordenadas relativas NED
        x_map_center = self.map_w_px / 2.0
        y_map_center = self.map_h_px / 2.0

        A = np.array([
            [0.0, -self.gsd, y_map_center * self.gsd - dy_m],
            [self.gsd, 0.0, -x_map_center * self.gsd - dx_m],
            [0.0, 0.0, max(10.0, altitude_agl_m)]
        ], dtype=np.float64)

        # Matriz proyectiva (Homografía) Map -> Camera
        M_map_to_cam = self.K @ (R_ned_c @ A)

        # Aplicar transformación de perspectiva
        rendered = cv2.warpPerspective(
            self.map_img,
            M_map_to_cam,
            (self.width, self.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        # 4. Simulación de efectos de sensor óptico real (ruido térmico gaussiano y viñeteado)
        if add_sensor_noise:
            noise = np.random.normal(0, 3.0, (self.height, self.width, 3))
            rendered = np.clip(rendered.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return rendered


class VirtualSkyCamera(PinholeCamera):
    """
    Cámara virtual cenital / Star Tracker (orientada al cielo).
    Simula la observación de constelaciones y estrellas para estimar la orientación 3D.
    """

    def __init__(self, width: int = 640, height: int = 480, fov_deg: float = 60.0):
        super().__init__(width, height, fov_deg)

        # Catálogo estelar sintético con las principales estrellas de navegación y estrellas de fondo
        self.stars = self._build_star_catalog()

        # Matriz de montaje Cenital en ejes cuerpo (Body -> Camera Optical Frame):
        # Mira hacia arriba (-Z_b)
        # Eje X_c a lo largo del ala derecha (+Y_b)
        # Eje Y_c a lo largo de la nariz (+X_b)
        self.R_b_c = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0]
        ], dtype=np.float64)

    def _build_star_catalog(self) -> List[Dict[str, Any]]:
        """Construye un catálogo de ~500 estrellas con coordenadas ecuatoriales (AR, Dec) y magnitudes."""
        np.random.seed(42)
        stars = []

        # Estrellas de navegación prominentes
        prominent = [
            ("Vega", 279.23, 38.78, 0.03),
            ("Arcturus", 213.91, 19.18, -0.05),
            ("Capella", 79.17, 45.99, 0.08),
            ("Rigel", 78.63, -8.20, 0.13),
            ("Betelgeuse", 88.79, 7.41, 0.50),
            ("Aldebaran", 68.98, 16.51, 0.85),
            ("Polaris", 37.95, 89.26, 1.98),
            ("Deneb", 310.35, 45.28, 1.25),
            ("Altair", 297.69, 8.87, 0.77),
            ("Antares", 247.35, -26.43, 1.06),
            ("Spica", 201.29, -11.16, 0.98),
            ("Pollux", 116.32, 28.02, 1.14),
        ]
        for name, ra_deg, dec_deg, mag in prominent:
            ra = np.radians(ra_deg)
            dec = np.radians(dec_deg)
            v_eci = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
            stars.append({"name": name, "v_eci": v_eci, "mag": mag})

        # Estrellas de fondo uniformemente distribuidas en la esfera celeste
        for _ in range(450):
            ra = np.random.uniform(0, 2 * np.pi)
            dec = np.arcsin(np.random.uniform(-1, 1))
            mag = np.random.uniform(2.5, 6.0)
            v_eci = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
            stars.append({"name": "", "v_eci": v_eci, "mag": mag})

        return stars

    def render_frame(
        self,
        timestamp_unix: float,
        lat_deg: float,
        lon_deg: float,
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
        add_sensor_noise: bool = True
    ) -> np.ndarray:
        """
        Renderiza la imagen del rastreador estelar (Star Tracker).
        """
        # Imagen base oscura (cielo nocturno con corriente oscura / dark noise)
        frame = np.full((self.height, self.width), 8, dtype=np.uint8)

        # 1. Rotación ECI -> ECEF -> NED
        # Tiempo sidéreo aparente de Greenwich simplificado (24h rotación)
        gst_rad = ((timestamp_unix % 86400.0) / 86400.0) * 2 * np.pi
        lon_rad = np.radians(lon_deg)
        lat_rad = np.radians(lat_deg)
        lst_rad = gst_rad + lon_rad  # Local Sidereal Time

        # Rotación ECI a local NED
        # v_ned = R_eci_ned * v_eci
        R_eci_ned = np.array([
            [-np.sin(lat_rad) * np.cos(lst_rad), -np.sin(lat_rad) * np.sin(lst_rad), np.cos(lat_rad)],
            [-np.sin(lst_rad), np.cos(lst_rad), 0.0],
            [-np.cos(lat_rad) * np.cos(lst_rad), -np.cos(lat_rad) * np.sin(lst_rad), -np.sin(lat_rad)]
        ])

        # 2. Rotación NED -> Body -> Camera
        phi = np.radians(roll_deg)
        theta = np.radians(pitch_deg)
        psi = np.radians(yaw_deg)

        R_x = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]])
        R_y = np.array([[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]])
        R_z = np.array([[np.cos(psi), -np.sin(psi), 0.0], [np.sin(psi), np.cos(psi), 0.0], [0.0, 0.0, 1.0]])
        R_ned_b = R_x @ R_y @ R_z

        R_eci_c = self.R_b_c @ (R_ned_b @ R_eci_ned)

        # 3. Proyectar estrellas en la cámara
        for star in self.stars:
            v_c = R_eci_c @ star["v_eci"]
            zc = v_c[2]

            # Debe estar frente a la lente (zc > 0)
            if zc <= 0.05:
                continue

            # Proyección Pinhole
            u = int(round(self.fx * (v_c[0] / zc) + self.cx))
            v = int(round(self.fy * (v_c[1] / zc) + self.cy))

            # Comprobar si cae dentro del sensor
            if 4 <= u < (self.width - 4) and 4 <= v < (self.height - 4):
                # Brillo estelar inversamente proporcional a la magnitud (Ley de Pogson)
                brightness = int(max(20, min(255, 255 * (10 ** (-0.4 * (star["mag"] - 0.5))))))
                radius = 1 if star["mag"] > 3.0 else 2

                # Núcleo brillante
                cv2.circle(frame, (u, v), radius, brightness, -1)
                # Halo difuso (PSF óptica)
                if star["mag"] <= 2.0:
                    cv2.circle(frame, (u, v), radius + 1, int(brightness * 0.45), 1)

        # 4. Ruido de lectura del sensor CMOS/CCD
        if add_sensor_noise:
            noise = np.random.normal(0, 4.0, (self.height, self.width))
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Convertir a 3 canales BGR para estandarizar formato
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
