"""
Detecta edificios en la imagen satelital del campus TecMM Zapopan
usando segmentación por color de techo (gris/beige vs verde/asfalto).
Convierte pixeles → coordenadas 3D de la escena Three.js.
"""
import json, math
from pathlib import Path

try:
    from PIL import Image, ImageFilter
    import numpy as np
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "Pillow", "numpy", "-q"])
    from PIL import Image, ImageFilter
    import numpy as np

ROOT = Path(__file__).resolve().parent.parent
img_path = ROOT / "data" / "tecmm_campus_map.png"
meta_path = ROOT / "data" / "tecmm_campus_meta.json"

with open(meta_path, encoding="utf-8") as f:
    meta = json.load(f)

CX_LAT = meta["center_lat"]
CX_LON = meta["center_lon"]
K_LAT  = meta["k_lat"]
K_LON  = meta["k_lon"]
W_M    = meta["width_m"]
H_M    = meta["height_m"]

img = Image.open(img_path).convert("RGB")
W_PX, H_PX = img.size   # 1536 x 1536
arr = np.array(img, dtype=np.float32)

print(f"Imagen: {W_PX}x{H_PX} px  |  Campus: {W_M:.1f}x{H_M:.1f} m")
print(f"Escala: {W_PX/W_M:.3f} px/m\n")

# ── Detección de edificios por color ──────────────────────────────────────────
# Los techos de los edificios en imágenes satelitales de Zoom 18 Esri suelen ser:
#   - Blanco/gris claro (R>160, G>160, B>160, baja saturación)
#   - Beige/ocre (R>150, G>120, B<100)
#   - Gris medio (R:100-170, G:100-170, B:100-170)
# El pasto es verde (G>R, G>B), el asfalto gris oscuro uniforme (R≈G≈B, <120)

R = arr[:,:,0]; G = arr[:,:,1]; B = arr[:,:,2]
# Saturación (max-min)/max
mx = np.maximum(np.maximum(R,G),B)
mn = np.minimum(np.minimum(R,G),B)
sat = np.where(mx>0, (mx-mn)/mx, 0)

# Edificio: brillo medio-alto, baja saturación (gris), NO es verde dominante
brightness = (R+G+B)/3.0
is_building = (
    (brightness > 130) &      # suficientemente claro
    (brightness < 240) &      # no es nube/sobre-exposición
    (sat < 0.25) &            # poca saturación (gris/beige)
    (G < R + 15) &            # no dominantemente verde (pasto)
    (B < R + 20)              # no dominantemente azul
)

mask = is_building.astype(np.uint8) * 255

# Erosión + dilatación para limpiar ruido
mask_img = Image.fromarray(mask)
# Suavizar con blur y volver a umbralizar
mask_blur = np.array(mask_img.filter(ImageFilter.GaussianBlur(3)))
mask_clean = (mask_blur > 100).astype(np.uint8)

# Usar scipy si disponible, si no hacemos blob manual
try:
    from scipy import ndimage
    labeled, n_blobs = ndimage.label(mask_clean)
    print(f"Blobs detectados: {n_blobs}")

    buildings = []
    PX_TO_M = W_M / W_PX   # metros por pixel

    for i in range(1, n_blobs+1):
        blob = (labeled == i)
        area_px = blob.sum()
        area_m2 = area_px * (PX_TO_M**2)

        # Filtrar: solo edificios reales (5m² a 5000m²)
        if area_m2 < 60 or area_m2 > 8000:
            continue

        # Centroide en pixeles
        rows, cols = np.where(blob)
        cx_px = cols.mean()
        cy_px = rows.mean()

        # Bounding box
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()
        w_px = c_max - c_min
        h_px = r_max - r_min
        w_m  = w_px * PX_TO_M
        h_m  = h_px * PX_TO_M

        # Convertir centroide pixel → coordenadas 3D de la escena Three.js
        # pixel(0,0) = esquina noroeste = lat_max, lon_min
        lon = meta["lon_min"] + (cx_px / W_PX) * (meta["lon_max"] - meta["lon_min"])
        lat = meta["lat_max"] - (cy_px / H_PX) * (meta["lat_max"] - meta["lat_min"])
        x3d = round((lon - CX_LON) * K_LON, 1)
        z3d = round(-(lat - CX_LAT) * K_LAT, 1)

        buildings.append({
            "id": i,
            "area_m2": round(area_m2, 1),
            "cx_px": round(cx_px, 1),
            "cy_px": round(cy_px, 1),
            "w_m": round(w_m, 1),
            "h_m": round(h_m, 1),
            "x3d": x3d,
            "z3d": z3d,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
        })

    # Ordenar por área descendente
    buildings.sort(key=lambda b: -b["area_m2"])

    print(f"\nEdificios (5–8000 m²): {len(buildings)} encontrados\n")
    print(f"{'ID':>4} {'Área m²':>8} {'W_m':>6} {'H_m':>6} {'X3D':>8} {'Z3D':>8}  lat/lon")
    print("-"*75)
    for b in buildings[:30]:
        print(f"{b['id']:4d}  {b['area_m2']:8.0f}  {b['w_m']:6.1f}  {b['h_m']:6.1f}  "
              f"{b['x3d']:+8.1f}  {b['z3d']:+8.1f}  {b['lat']},{b['lon']}")

    out_path = ROOT / "data" / "campus_buildings_detected.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(buildings[:40], f, indent=2)
    print(f"\nGuardado: {out_path}")

except ImportError:
    print("scipy no disponible — instalando...")
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "scipy", "-q"])
    print("Reinicia el script.")
