"""
clean.py — Limpia los archivos generados por el pipeline de simulacion.
Elimina datos, imagenes de camaras y graficas, manteniendo el codigo fuente intacto.

Uso:
    python src/clean.py           # pide confirmacion
    python src/clean.py --force   # elimina sin preguntar
"""

import argparse
import shutil
import sys
from pathlib import Path

# Directorio raiz del proyecto (padre de src/)
ROOT = Path(__file__).resolve().parent.parent

# ─── Archivos individuales a eliminar ───────────────────────────────────────
ARCHIVOS_A_ELIMINAR = [
    ROOT / "data" / "flight_log.csv",
    ROOT / "data" / "imu_measurements.csv",
    ROOT / "data" / "camera_ground_frames.csv",
    ROOT / "data" / "camera_sky_frames.csv",
    ROOT / "data" / "reference_map.png",
    ROOT / "data" / "reference_map_meta.json",
    ROOT / "data" / "visual_navigation_results.csv",
    ROOT / "data" / "fused_navigation_log.csv",
    ROOT / "data" / "physical_imu_log.csv",
    ROOT / "data" / "test_cam_view.png",
]

# ─── Carpetas completas a eliminar ──────────────────────────────────────────
CARPETAS_A_ELIMINAR = [
    ROOT / "data" / "camera_ground",
    ROOT / "data" / "camera_sky",
]

# ─── Carpeta cuyo contenido se elimina (pero no la carpeta en si) ───────────
CARPETAS_VACIADAS = [
    ROOT / "plots",
]


def obtener_tamano(ruta: Path) -> int:
    """Devuelve el tamano en bytes de un archivo o directorio."""
    if not ruta.exists():
        return 0
    if ruta.is_file():
        return ruta.stat().st_size
    total = 0
    for f in ruta.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def listar_objetivos() -> tuple[list[Path], list[Path], int]:
    """
    Devuelve (archivos_existentes, carpetas_existentes, bytes_totales).
    Para CARPETAS_VACIADAS, lista los archivos dentro de ellas.
    """
    archivos: list[Path] = []
    carpetas: list[Path] = []
    bytes_totales = 0

    for archivo in ARCHIVOS_A_ELIMINAR:
        if archivo.exists():
            archivos.append(archivo)
            bytes_totales += obtener_tamano(archivo)

    for carpeta in CARPETAS_A_ELIMINAR:
        if carpeta.exists():
            carpetas.append(carpeta)
            bytes_totales += obtener_tamano(carpeta)

    for carpeta in CARPETAS_VACIADAS:
        if carpeta.exists():
            for item in carpeta.iterdir():
                if item.is_file():
                    archivos.append(item)
                    bytes_totales += obtener_tamano(item)
                elif item.is_dir():
                    carpetas.append(item)
                    bytes_totales += obtener_tamano(item)

    return archivos, carpetas, bytes_totales


def eliminar_todo(archivos: list[Path], carpetas: list[Path]) -> tuple[int, int]:
    """
    Elimina los archivos y carpetas listados.
    Devuelve (num_archivos_eliminados, num_carpetas_eliminadas).
    """
    archivos_eliminados = 0
    carpetas_eliminadas = 0

    for archivo in archivos:
        try:
            archivo.unlink(missing_ok=True)
            archivos_eliminados += 1
        except Exception as e:
            print(f"  [AVISO] No se pudo eliminar {archivo.name}: {e}")

    for carpeta in carpetas:
        try:
            shutil.rmtree(carpeta, ignore_errors=False)
            carpetas_eliminadas += 1
        except Exception as e:
            print(f"  [AVISO] No se pudo eliminar carpeta {carpeta.name}: {e}")

    return archivos_eliminados, carpetas_eliminadas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Elimina archivos generados por el pipeline de simulacion."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Eliminar sin pedir confirmacion (para uso en automatizacion).",
    )
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════╗")
    print("║        LIMPIEZA DEL PROYECTO - SIMULACION        ║")
    print("╚══════════════════════════════════════════════════╝\n")

    archivos, carpetas, bytes_totales = listar_objetivos()

    if not archivos and not carpetas:
        print("  No hay archivos generados que limpiar. El proyecto ya esta limpio.")
        sys.exit(0)

    # ─── Mostrar lo que se va a eliminar ────────────────────────────────────
    print("  Archivos que se eliminaran:")
    for f in archivos:
        try:
            ruta_rel = f.relative_to(ROOT)
        except ValueError:
            ruta_rel = f
        size_kb = obtener_tamano(f) / 1024
        print(f"    - {ruta_rel}  ({size_kb:.1f} KB)")

    if carpetas:
        print("\n  Carpetas que se eliminaran (con todo su contenido):")
        for c in carpetas:
            try:
                ruta_rel = c.relative_to(ROOT)
            except ValueError:
                ruta_rel = c
            size_mb = obtener_tamano(c) / (1024 * 1024)
            print(f"    - {ruta_rel}/  ({size_mb:.2f} MB)")

    mb_totales = bytes_totales / (1024 * 1024)
    print(f"\n  Espacio aproximado a liberar: {mb_totales:.2f} MB")
    print("\n  Se conservan: scenarios/, src/, .venv/, README.md, .gitignore\n")

    # ─── Confirmacion ────────────────────────────────────────────────────────
    if not args.force:
        respuesta = input("  Eliminar estos archivos? [s/N]: ").strip().lower()
        if respuesta not in ("s",):
            print("\n  Operacion cancelada. No se elimino nada.")
            sys.exit(0)

    # ─── Eliminar ────────────────────────────────────────────────────────────
    print("\n  Eliminando...")
    num_archivos, num_carpetas = eliminar_todo(archivos, carpetas)

    print(f"\n  ✓ {num_archivos} archivo(s) eliminado(s).")
    if num_carpetas:
        print(f"  ✓ {num_carpetas} carpeta(s) eliminada(s).")
    print(f"  ✓ Espacio liberado: aproximadamente {mb_totales:.2f} MB")
    print("\n  Limpieza completada.\n")


if __name__ == "__main__":
    main()
