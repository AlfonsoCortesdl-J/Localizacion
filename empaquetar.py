#!/usr/bin/env python3
"""Crea un ZIP listo para llevar el proyecto a otra PC.

Incluye codigo, datos de mision, mapas y modelos 3D. Excluye el entorno
virtual, cache, graficas, sesiones y (opcionalmente) los fotogramas de las
camaras simuladas, que son los archivos mas pesados.

Uso:
  python empaquetar.py                  # paquete completo (~525 MB de datos)
  python empaquetar.py --sin-camaras    # sin data/camera_ground ni camera_sky
  python empaquetar.py --destino D:\\Localizacion.zip
"""

from __future__ import annotations

import argparse
import os
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".vscode",
    ".idea",
    "plots",
    "logs",
    "live_sessions",
    "paquete_avion_fisico",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".zip"}
CAMERA_DIRS = {"camera_ground", "camera_sky"}


def should_skip(path: Path, skip_cameras: bool) -> bool:
    parts = set(path.relative_to(ROOT).parts)
    if parts & EXCLUDE_DIRS:
        return True
    if skip_cameras and parts & CAMERA_DIRS:
        return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    return path.name in {"Thumbs.db", ".DS_Store"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Empaqueta el proyecto para otra PC.")
    parser.add_argument("--sin-camaras", action="store_true", help="excluir fotogramas simulados (~511 MB)")
    parser.add_argument("--destino", default=None, help="ruta del ZIP de salida")
    args = parser.parse_args()

    destination = Path(args.destino) if args.destino else ROOT / "Localizacion_portable.zip"
    destination = destination.resolve()
    if destination.exists():
        destination.unlink()

    files: list[Path] = []
    for base, dirs, names in os.walk(ROOT):
        base_path = Path(base)
        dirs[:] = [d for d in dirs if not should_skip(base_path / d, args.sin_camaras)]
        for name in names:
            path = base_path / name
            if path == destination:
                continue
            if should_skip(path, args.sin_camaras):
                continue
            files.append(path)

    total = sum(p.stat().st_size for p in files)
    print(f"Empaquetando {len(files)} archivos ({total / 1024 / 1024:.1f} MB)...")
    started = time.time()
    executable = {"iniciar.sh", "run.py", "empaquetar.py"}
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for index, path in enumerate(files, 1):
            name = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo.from_file(path, name)
            info.compress_type = zipfile.ZIP_DEFLATED
            # Conservar el bit de ejecucion para que iniciar.sh funcione al descomprimir.
            if path.name in executable:
                info.external_attr = (0o100755 << 16)
            with open(path, "rb") as source:
                zf.writestr(info, source.read(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
            if index % 200 == 0:
                print(f"  {index}/{len(files)}...", flush=True)

    size = destination.stat().st_size / 1024 / 1024
    print(f"Listo: {destination}  ({size:.1f} MB, {time.time() - started:.0f} s)")
    print("En la otra PC: descomprime y ejecuta iniciar.bat (Windows) o iniciar.sh (Linux/macOS).")


if __name__ == "__main__":
    main()
