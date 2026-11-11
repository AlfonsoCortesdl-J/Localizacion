"""Convert a north-up, projected GeoTIFF orthomosaic to the map format used here.

Example from the project root:
  python src/prepare_reference_map.py --geotiff maps/orthomosaic_utm.tif

Optional semantic labels must be a single-band GeoTIFF with identical grid,
CRS, and dimensions. Without labels, the output is a georeferenced visual map
for ORB/map matching only; it is not a segmentation training mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def _to_uint8(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        out = rgb.copy()
    else:
        out = np.zeros(rgb.shape, dtype=np.uint8)
        for band in range(rgb.shape[2]):
            values = rgb[:, :, band][valid]
            if values.size == 0:
                raise ValueError("El GeoTIFF no contiene píxeles válidos.")
            low, high = np.percentile(values, (2, 98))
            if high <= low:
                high = low + 1.0
            scaled = (rgb[:, :, band].astype(np.float32) - low) * (255.0 / (high - low))
            out[:, :, band] = np.clip(scaled, 0, 255).astype(np.uint8)
    out[~valid] = 0
    return out


def convert(
    geotiff_path: Path,
    output_dir: Path,
    labels_path: Optional[Path] = None,
    class_map_path: Optional[Path] = None,
) -> Tuple[Path, Path]:
    try:
        import rasterio
        from pyproj import CRS, Transformer
    except ImportError as exc:
        raise RuntimeError("Falta rasterio o pyproj. Instálalos con: python -m pip install rasterio pyproj") from exc

    with rasterio.open(geotiff_path) as src:
        if src.crs is None:
            raise ValueError("El GeoTIFF no tiene CRS/proyección; expórtalo georreferenciado desde QGIS/GIS.")
        crs = CRS.from_user_input(src.crs)
        if not crs.is_projected:
            raise ValueError("Se requiere un mapa reproyectado a un CRS métrico, normalmente UTM; el CRS actual es geográfico.")
        unit = (crs.axis_info[0].unit_name or "").lower()
        if unit not in {"metre", "meter", "metres", "meters"}:
            raise ValueError(f"El CRS debe usar metros; unidad detectada: {unit or 'desconocida'}")

        transform = src.transform
        if abs(transform.b) > 1e-9 or abs(transform.d) > 1e-9 or transform.a <= 0 or transform.e >= 0:
            raise ValueError("El raster debe estar orientado norte-arriba y sin rotación. Reproyéctalo antes de convertirlo.")
        gsd_x, gsd_y = abs(float(transform.a)), abs(float(transform.e))
        if abs(gsd_x - gsd_y) / max(gsd_x, gsd_y) > 0.02:
            raise ValueError("MapMatcher supone píxeles cuadrados; reproyecta/remuestrea el GeoTIFF a GSD uniforme.")

        if src.count >= 3:
            rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        else:
            band = src.read(1)
            rgb = np.repeat(band[:, :, None], 3, axis=2)
        valid = src.dataset_mask() > 0
        rgb = _to_uint8(rgb, valid)
        height, width = rgb.shape[:2]
        center_x, center_y = transform * (width / 2.0, height / 2.0)
        bounds = src.bounds
        source_crs = str(src.crs)

        to_wgs84 = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
        center_lon, center_lat = to_wgs84.transform(center_x, center_y)
        corners = [
            to_wgs84.transform(bounds.left, bounds.bottom),
            to_wgs84.transform(bounds.left, bounds.top),
            to_wgs84.transform(bounds.right, bounds.bottom),
            to_wgs84.transform(bounds.right, bounds.top),
        ]

        label_array = None
        if labels_path:
            with rasterio.open(labels_path) as label_src:
                same_grid = (
                    label_src.width == src.width
                    and label_src.height == src.height
                    and label_src.crs == src.crs
                    and label_src.transform == src.transform
                )
                if not same_grid:
                    raise ValueError("La máscara debe coincidir exactamente con el GeoTIFF: tamaño, CRS y transform.")
                if label_src.count != 1:
                    raise ValueError("La máscara semántica debe ser de una sola banda con IDs de clase por píxel.")
                raw_labels = label_src.read(1)
                if raw_labels.min() < 0 or raw_labels.max() > 255:
                    raise ValueError("Los IDs de clase de la máscara deben estar en el rango 0..255.")
                label_array = raw_labels.astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = output_dir / "reference_map.png"
    meta_path = output_dir / "reference_map_meta.json"
    if not cv2.imwrite(str(map_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"No se pudo guardar {map_path}")

    metadata = {
        "map_filename": map_path.name,
        "format": "RGB",
        "width_px": int(width),
        "height_px": int(height),
        "width_m": float(width * gsd_x),
        "height_m": float(height * gsd_y),
        "gsd_m_per_px": float((gsd_x + gsd_y) / 2.0),
        "source_crs": source_crs,
        "center_projected_m": {"easting": float(center_x), "northing": float(center_y)},
        "center_geodetic": {"latitude_deg": float(center_lat), "longitude_deg": float(center_lon)},
        "bounding_box_wgs84": {
            "lat_min": float(min(lat for lon, lat in corners)),
            "lat_max": float(max(lat for lon, lat in corners)),
            "lon_min": float(min(lon for lon, lat in corners)),
            "lon_max": float(max(lon for lon, lat in corners)),
        },
        "map_source": str(geotiff_path.resolve()),
    }
    if label_array is not None:
        labels_out = output_dir / "reference_map_labels.png"
        if not cv2.imwrite(str(labels_out), label_array):
            raise RuntimeError(f"No se pudo guardar {labels_out}")
        metadata["labels_filename"] = labels_out.name
        if class_map_path:
            metadata["class_ids"] = json.loads(class_map_path.read_text(encoding="utf-8"))

    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Mapa convertido: {map_path}")
    print(f"Metadatos geográficos: {meta_path}")
    if label_array is None:
        print("No se generó máscara semántica: este mapa sirve para matching visual, no para entrenar segmentación.")
    else:
        print(f"Máscara semántica: {output_dir / 'reference_map_labels.png'}")
    return map_path, meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara un ortomosaico GeoTIFF para MapMatcher.")
    parser.add_argument("--geotiff", required=True, help="Ortomosaico en GeoTIFF con CRS proyectado en metros.")
    parser.add_argument("--output-dir", default="data/real_map", help="Salida; por defecto data/real_map.")
    parser.add_argument("--labels", help="GeoTIFF opcional de una banda con IDs semánticos y la misma cuadrícula.")
    parser.add_argument("--class-map", help="JSON opcional que describe los IDs guardados en --labels.")
    args = parser.parse_args()
    convert(
        Path(args.geotiff),
        Path(args.output_dir),
        Path(args.labels) if args.labels else None,
        Path(args.class_map) if args.class_map else None,
    )


if __name__ == "__main__":
    main()
