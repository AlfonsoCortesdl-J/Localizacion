"""
run_all.py — Orquestador del pipeline de simulacion de vuelo.
Ejecuta los 7 pasos en orden, mide tiempos y muestra un resumen final.
"""

import subprocess
import sys
import time
from pathlib import Path

# Configurar salida de texto robusta para Windows PowerShell
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Directorio raiz del proyecto (padre de src/)
ROOT = Path(__file__).resolve().parent.parent

# Banner ASCII compatible con todas las terminales
BANNER = r"""
================================================================
  SIMULACION DE VUELO Y NAVEGACION AEREA AUTONOMA - CESSNA 172
================================================================
  JSBSim 6-DOF + Camaras Virtuales + Odometria Visual + EKF
================================================================
"""

PASOS = [
    {
        "numero": 1,
        "descripcion": "Simulando 150 segundos de vuelo del Cessna 172 (circuito oval)...",
        "nombre_corto": "Simulacion de vuelo",
        "script": "src/record_flight.py",
    },
    {
        "numero": 2,
        "descripcion": "Generando graficas de dinamica de vuelo...",
        "nombre_corto": "Graficas de vuelo",
        "script": "src/plot_results.py",
    },
    {
        "numero": 3,
        "descripcion": "Creando el mapa de referencia del terreno...",
        "nombre_corto": "Mapa de referencia",
        "script": "src/generate_reference_map.py",
    },
    {
        "numero": 4,
        "descripcion": "Renderizando camaras virtuales (tarda ~1-2 min)...",
        "nombre_corto": "Camaras virtuales",
        "script": "src/render_virtual_cameras.py",
    },
    {
        "numero": 5,
        "descripcion": "Ejecutando algoritmo de navegacion visual (VO+TRN)...",
        "nombre_corto": "Navegacion visual",
        "script": "src/visual_navigation.py",
    },
    {
        "numero": 6,
        "descripcion": "Fusionando sensores con filtro de Kalman (EKF)...",
        "nombre_corto": "Fusion EKF",
        "script": "src/sensor_fusion_ekf.py",
    },
    {
        "numero": 7,
        "descripcion": "Generando panel didactico para presentacion...",
        "nombre_corto": "Panel de presentacion",
        "script": "src/plot_presentation.py",
    },
]


def ejecutar_paso(paso: dict) -> float:
    """Ejecuta un paso del pipeline y devuelve el tiempo transcurrido en segundos."""
    num = paso["numero"]
    desc = paso["descripcion"]
    script = paso["script"]

    print(f"\n{'-' * 60}")
    print(f"  Paso {num}/7: {desc}")
    print(f"{'-' * 60}")

    inicio = time.perf_counter()
    resultado = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    fin = time.perf_counter()
    elapsed = fin - inicio

    if resultado.stdout.strip():
        for linea in resultado.stdout.strip().splitlines():
            print(f"  | {linea}")

    if resultado.returncode != 0:
        print(f"\n  [ERROR] en Paso {num}:")
        if resultado.stderr.strip():
            for linea in resultado.stderr.strip().splitlines():
                print(f"  ! {linea}")
        print(f"\n  El proceso termino con codigo {resultado.returncode}.")
        print(f"  Deteniendo el pipeline en el Paso {num}.")
        sys.exit(1)

    print(f"\n  [OK] Paso {num} completado en {elapsed:.1f} s")
    return elapsed


def imprimir_resumen(tiempos: list[float]) -> None:
    """Imprime la caja de resumen final con todos los tiempos."""
    total = sum(tiempos)
    ancho = 56

    linea_borde = "=" * ancho
    linea_sep = "-" * ancho

    print(f"\n{linea_borde}")
    print("   SIMULACION Y CALCULOS COMPLETADOS CON EXITO")
    print(linea_borde)

    for paso, elapsed in zip(PASOS, tiempos):
        nombre = paso["nombre_corto"]
        tiempo_str = f"{elapsed:.1f} s"
        print(f"  Paso {paso['numero']}: {nombre:<32} {tiempo_str:>8}")

    print(f"  {linea_sep}")
    total_str = f"{total:.1f} s"
    print(f"  TOTAL:{'':>35} {total_str:>8}")
    print(linea_borde)
    print("  Resultados listos en:")
    print("    - data/flight_log.csv")
    print("    - data/fused_navigation_log.csv")
    print("    - plots/presentacion.png   <-- Panel resumen para diapositivas")
    print("")
    print("  Para ver la simulacion animada en vivo con mapa e instrumentos:")
    print("    python src/live_simulation.py")
    print(linea_borde)


def main() -> None:
    print(BANNER)
    print(f"  Directorio raiz: {ROOT}")
    print(f"  Python:          {sys.executable}")

    tiempos: list[float] = []

    for paso in PASOS:
        elapsed = ejecutar_paso(paso)
        tiempos.append(elapsed)

    imprimir_resumen(tiempos)


if __name__ == "__main__":
    main()
