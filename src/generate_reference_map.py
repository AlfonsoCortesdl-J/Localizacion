"""
generate_reference_map.py
=========================
Genera un ortomapa georreferenciado sintético de alta resolución (2000x2000 píxeles,
GSD = 2.0 m/px) centrado en la zona de vuelo.

Incluye características realistas del terreno para algoritmos de visión artificial:
- Red de carreteras asfaltadas y caminos de terracería con intersecciones.
- Pista de aterrizaje con marcas aeronáuticas y calles de rodaje.
- Parcelas agrícolas con texturas y reflectancias diferenciadas.
- Río con meandros y vegetación ripícola.
- Bloques urbanos y hangares con bordes contrastados.

Archivos generados:
-------------------
1. data/reference_map.png: Imagen georreferenciada en formato RGB.
2. data/reference_map_meta.json: Parámetros geodésicos y de escala del mapa.
"""

import os
import json
import numpy as np
import cv2

# Parámetros geodésicos estándar WGS84
METERS_PER_DEG_LAT = 111320.0


def lat_lon_to_meters(lat: float, lon: float, center_lat: float, center_lon: float) -> tuple[float, float]:
    """Convierte coordenadas geodésicas (lat, lon) a coordenadas métricas locales (dx, dy) respecto al centro."""
    meters_per_deg_lon = METERS_PER_DEG_LAT * np.cos(np.radians(center_lat))
    dx = (lon - center_lon) * meters_per_deg_lon  # Este (+)
    dy = (lat - center_lat) * METERS_PER_DEG_LAT  # Norte (+)
    return dx, dy


def meters_to_lat_lon(dx: float, dy: float, center_lat: float, center_lon: float) -> tuple[float, float]:
    """Convierte coordenadas métricas locales (dx, dy) a coordenadas geodésicas (lat, lon)."""
    meters_per_deg_lon = METERS_PER_DEG_LAT * np.cos(np.radians(center_lat))
    lat = center_lat + dy / METERS_PER_DEG_LAT
    lon = center_lon + dx / meters_per_deg_lon
    return lat, lon


def create_synthetic_reference_map(
    output_dir: str = "data",
    center_lat: float = 20.7768,
    center_lon: float = -103.3365,
    map_size_m: float = 8000.0,
    map_size_px: int = 2000,
) -> tuple[str, str]:
    """Genera la imagen del mapa georreferenciado y su archivo de metadatos JSON."""
    os.makedirs(output_dir, exist_ok=True)
    gsd = map_size_m / map_size_px  # metros por píxel (2.0 m/px)

    print(f"Generando ortomapa sintético ({map_size_px}x{map_size_px} px, GSD = {gsd:.2f} m/px, cobertura = {map_size_m/1000:.1f}x{map_size_m/1000:.1f} km)...")

    # Lienzo base: Suelo con tono tierra/pastizal y ruido Perlin/Gaussiano sutil
    np.random.seed(42)  # Reproducibilidad exacta
    base_color = np.array([130, 165, 140], dtype=np.uint8)  # BGR: tono verde olivo/pasto seco
    img = np.full((map_size_px, map_size_px, 3), base_color, dtype=np.uint8)

    # Añadir variaciones tonales de terreno a gran escala
    noise = np.random.normal(0, 8, (map_size_px // 10, map_size_px // 10, 3))
    noise = cv2.resize(noise, (map_size_px, map_size_px), interpolation=cv2.INTER_CUBIC)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 1. Parcelas agrícolas con diferentes tonalidades (cultivos, tierra arada)
    grid_steps = np.linspace(100, map_size_px - 100, 12, dtype=int)
    for i in range(len(grid_steps) - 1):
        for j in range(len(grid_steps) - 1):
            x1, x2 = grid_steps[i], grid_steps[i + 1]
            y1, y2 = grid_steps[j], grid_steps[j + 1]

            # Variedad de tonos agrícolas: verde oscuro, trigo dorado, tierra arada
            crop_type = (i * 7 + j * 13) % 4
            if crop_type == 0:
                color = (70 + (i * 3) % 20, 130 + (j * 4) % 30, 60 + (i * 5) % 20)  # Cultivo verde
            elif crop_type == 1:
                color = (80 + (j * 4) % 15, 150 + (i * 5) % 25, 180 + (j * 3) % 30)  # Trigo / pasto seco
            elif crop_type == 2:
                color = (65 + (i * 4) % 15, 95 + (j * 3) % 20, 135 + (i * 4) % 25)  # Tierra marrón
            else:
                color = (90 + (j * 5) % 20, 140 + (i * 3) % 25, 110 + (j * 2) % 20)  # Matorral

            cv2.rectangle(img, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), color, -1)

            # Surcos o patrones de arado en algunas parcelas
            if (i + j) % 3 == 0:
                for line_y in range(y1 + 8, y2 - 8, 12):
                    cv2.line(img, (x1 + 6, line_y), (x2 - 6, line_y), (color[0] - 15, color[1] - 15, color[2] - 15), 1)

    # 2. Río con meandros naturales y vegetación
    river_pts = []
    for y in range(0, map_size_px, 20):
        # Función senoidal con armónicos para simular curvatura natural
        x = int(600 + 180 * np.sin(y * 0.004) + 60 * np.cos(y * 0.012))
        river_pts.append([x, y])
    river_pts = np.array(river_pts, dtype=np.int32)
    # Ribera con vegetación (tono verde oscuro)
    cv2.polylines(img, [river_pts], False, (40, 80, 30), thickness=28)
    # Agua del río (azul-grisáceo)
    cv2.polylines(img, [river_pts], False, (140, 110, 60), thickness=16)

    # 3. Pista de aterrizaje aeronáutica (Aeródromo de prueba)
    rwy_center_x, rwy_center_y = 1350, 950
    rwy_length_px = int(1400 / gsd)  # 1400 m de pista = 700 px
    rwy_width_px = int(45 / gsd)     # 45 m de ancho = 23 px
    angle_deg = 20.0  # Orientación aproximada 02/20
    rect = ((rwy_center_x, rwy_center_y), (rwy_width_px, rwy_length_px), angle_deg)
    box = cv2.boxPoints(rect).astype(np.int32)

    # Asfalto de la pista
    cv2.drawContours(img, [box], 0, (65, 65, 65), -1)
    # Borde de la pista
    cv2.drawContours(img, [box], 0, (90, 90, 90), 2)

    # Línea central punteada blanca de la pista
    rad = np.radians(angle_deg)
    dir_vec = np.array([-np.sin(rad), np.cos(rad)])
    for step in range(-rwy_length_px // 2 + 30, rwy_length_px // 2 - 30, 40):
        p1 = (int(rwy_center_x + dir_vec[0] * step), int(rwy_center_y + dir_vec[1] * step))
        p2 = (int(rwy_center_x + dir_vec[0] * (step + 20)), int(rwy_center_y + dir_vec[1] * (step + 20)))
        cv2.line(img, p1, p2, (230, 230, 230), 2)

    # 4. Red de carreteras principales y secundarias
    # Carretera Federal Este-Oeste (doble carril asfaltado)
    hwy_y = 1250
    cv2.line(img, (0, hwy_y), (map_size_px, hwy_y), (60, 60, 60), thickness=14)
    cv2.line(img, (0, hwy_y), (map_size_px, hwy_y), (220, 220, 220), thickness=2)  # Línea central amarilla/blanca

    # Carretera Norte-Sur con intersección y glorieta
    hwy_x = 1050
    cv2.line(img, (hwy_x, 0), (hwy_x, map_size_px), (60, 60, 60), thickness=12)
    cv2.line(img, (hwy_x, 0), (hwy_x, map_size_px), (220, 220, 220), thickness=1)

    # Glorieta en la intersección
    cv2.circle(img, (hwy_x, hwy_y), 32, (60, 60, 60), -1)
    cv2.circle(img, (hwy_x, hwy_y), 16, (40, 110, 50), -1)  # Isla verde central

    # 5. Complejo de hangares y naves industriales (geometrías de alto contraste para matching visual)
    hangar_coords = [(1150, 850), (1220, 850), (1150, 930), (1220, 930)]
    for hx, hy in hangar_coords:
        # Techo de hangar metálico blanco/gris claro con sombra
        cv2.rectangle(img, (hx + 4, hy + 4), (hx + 45, hy + 35), (40, 40, 40), -1)  # Sombra
        cv2.rectangle(img, (hx, hy), (hx + 40, hy + 30), (210, 215, 220), -1)        # Techo

    # Guardar imagen PNG del mapa
    map_image_path = os.path.join(output_dir, "reference_map.png")
    cv2.imwrite(map_image_path, img)
    print(f"  [OK] Imagen del mapa guardada en: {map_image_path} ({os.path.getsize(map_image_path)/1024/1024:.2f} MB)")

    # Calcular límites geodésicos WGS84
    meters_per_deg_lon = METERS_PER_DEG_LAT * np.cos(np.radians(center_lat))
    half_span_m = map_size_m / 2.0
    lat_min = center_lat - (half_span_m / METERS_PER_DEG_LAT)
    lat_max = center_lat + (half_span_m / METERS_PER_DEG_LAT)
    lon_min = center_lon - (half_span_m / meters_per_deg_lon)
    lon_max = center_lon + (half_span_m / meters_per_deg_lon)

    metadata = {
        "map_filename": "reference_map.png",
        "format": "RGB",
        "width_px": map_size_px,
        "height_px": map_size_px,
        "width_m": map_size_m,
        "height_m": map_size_m,
        "gsd_m_per_px": gsd,
        "center_geodetic": {
            "latitude_deg": center_lat,
            "longitude_deg": center_lon
        },
        "bounding_box_wgs84": {
            "lat_min": round(lat_min, 6),
            "lat_max": round(lat_max, 6),
            "lon_min": round(lon_min, 6),
            "lon_max": round(lon_max, 6)
        },
        "landmarks": {
            "airfield_runway": {"center_lat": 20.7780, "center_lon": -103.3330, "heading_deg": 20.0},
            "highway_junction": {"lat": 20.7760, "lon": -103.3360},
            "river_bridge": {"lat": 20.7760, "lon": -103.3410}
        }
    }

    meta_json_path = os.path.join(output_dir, "reference_map_meta.json")
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  [OK] Metadatos geodésicos guardados en: {meta_json_path}")
    return map_image_path, meta_json_path


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_path = os.path.join(parent_dir, "data")
    create_synthetic_reference_map(data_path)
