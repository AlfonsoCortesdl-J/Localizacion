"""
fetch_tecmm_campus.py
=====================
Descarga la ortofoto satelital de alta resolución (Zoom 18) del
Tecnológico Superior de Jalisco — Campus Zapopan (TecMM Zapopan).
Rejilla de 6x6 tiles (1536 x 1536 px, ~860m x 855m) perfectamente centrada.
"""

import math
import json
import urllib.request
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CENTER_LAT = 20.702277
CENTER_LON = -103.473404
ZOOM = 18

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

def main():
    cx, cy = deg2num(CENTER_LAT, CENTER_LON, ZOOM)
    print(f"[Campus Map] Centro tile Z{ZOOM}: X={cx}, Y={cy}")

    radius = 3  # 6x6 tiles
    x_min = cx - radius
    x_max = cx + radius
    y_min = cy - radius
    y_max = cy + radius

    tile_w = 256
    tile_h = 256
    total_w = (x_max - x_min) * tile_w
    total_h = (y_max - y_min) * tile_h

    stitched = Image.new("RGB", (total_w, total_h))
    print(f"[Campus Map] Descargando { (x_max-x_min)*(y_max-y_min) } tiles satelitales...")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TecMMCampusMapper/1.0"}

    for ix, tx in enumerate(range(x_min, x_max)):
        for iy, ty in enumerate(range(y_min, y_max)):
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{ZOOM}/{ty}/{tx}"
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    from io import BytesIO
                    tile_img = Image.open(BytesIO(resp.read()))
                    stitched.paste(tile_img, (ix * tile_w, iy * tile_h))
            except Exception as e:
                print(f"  Aviso: Error descargando tile ({tx}, {ty}): {e}")

    nw_lat, nw_lon = num2deg(x_min, y_min, ZOOM)
    se_lat, se_lon = num2deg(x_max, y_max, ZOOM)

    lat_mid = (nw_lat + se_lat) / 2.0
    k_lat = 111132.954 - 559.822 * math.cos(2 * math.radians(lat_mid))
    k_lon = 111412.84 * math.cos(math.radians(lat_mid))

    width_m = abs(se_lon - nw_lon) * k_lon
    height_m = abs(nw_lat - se_lat) * k_lat

    out_png = DATA_DIR / "tecmm_campus_map.png"
    stitched.save(out_png, "PNG", optimize=True)
    print(f"[Campus Map] Ortomapa satelital guardado en: {out_png}")
    print(f"  Dimensiones en píxeles: {total_w} x {total_h}")
    print(f"  Dimensiones en metros: {width_m:.1f} m x {height_m:.1f} m")

    meta = {
        "institution": "Instituto Tecnológico José Mario Molina Pasquel y Henríquez — Campus Zapopan",
        "alias": "TecMM Zapopan / ITS Zapopan",
        "location": "Camino Arenero #1101, Col. El Bajío, Zapopan, Jalisco",
        "center_lat": (nw_lat + se_lat) / 2.0,
        "center_lon": (nw_lon + se_lon) / 2.0,
        "lat_min": se_lat,
        "lat_max": nw_lat,
        "lon_min": nw_lon,
        "lon_max": se_lon,
        "width_m": width_m,
        "height_m": height_m,
        "k_lat": k_lat,
        "k_lon": k_lon,
        "zoom": ZOOM
    }

    out_meta = DATA_DIR / "tecmm_campus_meta.json"
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[Campus Map] Metadatos actualizados en: {out_meta}")

if __name__ == "__main__":
    main()
