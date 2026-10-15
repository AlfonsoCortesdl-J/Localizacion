#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Ejecuta el pipeline completo de Simulación de Vuelo
# (Cessna 172P · JSBSim · VO/TRN · EKF · IMU STM32) descrito en el README.
#
# Uso (desde la raíz del proyecto, p. ej. en Git Bash / WSL / Linux / macOS):
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh                 # pipeline completo
#   ./run_pipeline.sh --skip-install  # no reinstalar dependencias
#   ./run_pipeline.sh --from 5        # reanudar desde el paso N (1-7)
#   ./run_pipeline.sh --clean         # borrar data/ y plots/ antes de correr
#
# Los logs de cada paso quedan en logs/paso_N.log
# =============================================================================
set -u -o pipefail

# ---------------------------- Configuración ----------------------------------
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SKIP_INSTALL=0
FROM_STEP=1
CLEAN=0
IMU_SIM_DURATION=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-install) SKIP_INSTALL=1 ;;
    --from)         FROM_STEP="${2:-1}"; shift ;;
    --clean)        CLEAN=1 ;;
    -h|--help)      sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "Opción desconocida: $1"; exit 2 ;;
  esac
  shift
done

cd "$PROJECT_DIR" || { echo "No existe $PROJECT_DIR"; exit 1; }
mkdir -p logs data plots

# ---------------------------- Utilidades -------------------------------------
if [[ -t 1 ]]; then G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; B=$'\e[34m'; N=$'\e[0m'; else G=; R=; Y=; B=; N=; fi
PASS=0; FAIL=0; WARN=0
declare -a SUMMARY=()

ok()   { echo "${G}[OK]${N}   $*";  PASS=$((PASS+1)); SUMMARY+=("OK    $*"); }
bad()  { echo "${R}[FAIL]${N} $*";  FAIL=$((FAIL+1)); SUMMARY+=("FAIL  $*"); }
warn() { echo "${Y}[WARN]${N} $*";  WARN=$((WARN+1)); SUMMARY+=("WARN  $*"); }
hdr()  { echo; echo "${B}=== $* ===${N}"; }

# Ejecuta un script Python, mide tiempo y guarda log. Aborta si falla.
run_step() {
  local n="$1" desc="$2"; shift 2
  if (( n < FROM_STEP )); then echo "(saltando paso $n)"; return 0; fi
  hdr "Paso $n · $desc"
  local t0=$SECONDS
  if "$@" 2>&1 | tee "logs/paso_${n}.log"; then
    ok "Paso $n ($desc) terminó en $((SECONDS - t0)) s"
  else
    bad "Paso $n ($desc) falló — revisa logs/paso_${n}.log"
    print_summary; exit 1
  fi
}

# Verifica que existan archivos y no estén vacíos
check_files() {
  local f
  for f in "$@"; do
    if [[ -s "$f" ]]; then ok "Existe $f"; else bad "Falta o está vacío: $f"; fi
  done
}

print_summary() {
  hdr "Resumen"
  printf '%s\n' "${SUMMARY[@]}"
  echo "-----------------------------------------"
  echo "OK: $PASS · WARN: $WARN · FAIL: $FAIL"
}

# ---------------------------- Entorno Python ---------------------------------
hdr "Entorno Python"
if [[ ! -d .venv ]]; then
  PY=$(command -v python3 || command -v python) || { echo "Python no encontrado"; exit 1; }
  "$PY" -m venv .venv || { echo "No se pudo crear .venv"; exit 1; }
fi

# Windows (Git Bash) usa Scripts/, Linux/macOS usa bin/
if   [[ -f .venv/Scripts/activate ]]; then source .venv/Scripts/activate
elif [[ -f .venv/bin/activate     ]]; then source .venv/bin/activate
else echo "No se encontró el script de activación de .venv"; exit 1; fi

PYTHON=python
"$PYTHON" --version

if (( SKIP_INSTALL == 0 )); then
  "$PYTHON" -m pip install --quiet jsbsim numpy pandas matplotlib opencv-python pyserial \
    || { echo "Falló pip install"; exit 1; }
fi

"$PYTHON" -c "import jsbsim; print('JSBSim OK —', jsbsim.__version__)" || { bad "JSBSim no disponible"; exit 1; }
"$PYTHON" -c "import cv2;    print('OpenCV OK —', cv2.__version__)"    || { bad "OpenCV no disponible"; exit 1; }
export MPLBACKEND=Agg   # gráficas sin ventana

# Comprobar que existan los scripts del README
for s in record_flight plot_results generate_reference_map render_virtual_cameras \
         visual_navigation sensor_fusion_ekf serial_imu_reader; do
  [[ -f "src/$s.py" ]] || { bad "Falta src/$s.py"; print_summary; exit 1; }
done
[[ -f scenarios/escenario_base.json ]] || { bad "Falta scenarios/escenario_base.json"; exit 1; }

if (( CLEAN == 1 && FROM_STEP == 1 )); then
  hdr "Limpieza"
  rm -rf data/* plots/* logs/*
fi

# ---------------------------- Pipeline ---------------------------------------
run_step 1 "Simulación y Ground Truth"       "$PYTHON" src/record_flight.py
run_step 2 "Gráficas de dinámica de vuelo"   "$PYTHON" src/plot_results.py
run_step 3 "Ortomapa de referencia"          "$PYTHON" src/generate_reference_map.py
run_step 4 "Cámaras virtuales"               "$PYTHON" src/render_virtual_cameras.py
run_step 5 "Navegación visual (VO + TRN)"    "$PYTHON" src/visual_navigation.py
run_step 6 "Fusión sensorial EKF"            "$PYTHON" src/sensor_fusion_ekf.py
run_step 7 "IMU física STM32 (modo simulado)" "$PYTHON" src/serial_imu_reader.py --simulate --duration "$IMU_SIM_DURATION"

# ---------------------------- Verificación -----------------------------------
hdr "Verificación de archivos generados"
check_files data/flight_log.csv data/imu_measurements.csv \
            data/camera_ground_frames.csv data/camera_sky_frames.csv \
            data/reference_map.png data/reference_map_meta.json \
            data/visual_navigation_results.csv data/fused_navigation_log.csv \
            data/physical_imu_log.csv

n_ground=$(find data/camera_ground -name '*.png' 2>/dev/null | wc -l)
n_sky=$(find data/camera_sky -name '*.png' 2>/dev/null | wc -l)
[[ "$n_ground" -eq 300 ]] && ok "300 fotogramas nadir" || bad "Fotogramas nadir: $n_ground (esperado 300)"
[[ "$n_sky"    -eq 120 ]] && ok "120 fotogramas sky"   || bad "Fotogramas sky: $n_sky (esperado 120)"

n_plots=$(find plots -maxdepth 1 -name '*.png' | wc -l)
[[ "$n_plots" -eq 10 ]] && ok "10 gráficas generadas" || bad "Gráficas: $n_plots (esperado 10)"

small=$(find plots -maxdepth 1 -name '*.png' -size -50k | wc -l)
[[ "$small" -eq 0 ]] && ok "Todas las gráficas > 50 kB" || warn "$small gráficas pesan < 50 kB (posible render vacío)"

hdr "Integridad del Ground Truth y de la IMU"
"$PYTHON" - <<'PYEOF'
import sys, pandas as pd, numpy as np
rc = 0
def chk(cond, msg):
    global rc
    print(("[OK]   " if cond else "[FAIL] ") + msg)
    if not cond: rc = 1

gt = pd.read_csv("data/flight_log.csv")
tc = next(c for c in gt.columns if c.lower().startswith("sim_time"))
print(f"Muestras: {len(gt)} | Duración: {gt[tc].max():.1f} s | "
      f"Altitud min/max: {gt.altitude_msl_m.min():.0f}/{gt.altitude_msl_m.max():.0f} m | "
      f"Nulos: {int(gt.isnull().sum().sum())}")
chk(len(gt) == 1200,                    "flight_log tiene 1200 muestras")
chk(abs(gt[tc].max() - 59.9) < 0.2, "duración ≈ 59.9 s")
chk(gt.isnull().sum().sum() == 0,       "sin valores nulos en Ground Truth")
chk(np.all(np.diff(gt[tc]) > 0), "tiempo estrictamente creciente")

imu = pd.read_csv("data/imu_measurements.csv")
ic = next((c for c in imu.columns if "time" in c.lower() or c.lower() in ("t","timestamp")), None)
dt = float(np.median(np.diff(imu[ic]))) if ic else float("nan")
print(f"[INFO] IMU: {len(imu)} filas, dt mediano = {dt:.4f} s (~{1/dt:.0f} Hz)" if ic else f"[INFO] IMU: {len(imu)} filas")
chk(imu.isnull().sum().sum() == 0, "sin nulos en IMU simulada")

# Consistencia física: en crucero la fuerza específica en Z ≈ -g (marco FRD)
az = gt.accel_specific_z_mps2.iloc[:150].mean()   # primeros 7.5 s, vuelo nivelado
chk(-11.0 < az < -8.5, f"fuerza específica Z en nivelado ≈ -9.81 (medido {az:.2f})")

# El viraje de 15–35 s debe mostrar alabeo cercano a +20°
roll = gt[(gt[tc] > 25) & (gt[tc] < 35)].roll_deg.mean()
chk(10 < roll < 30, f"alabeo medio en viraje ≈ 20° (medido {roll:.1f}°)")

sys.exit(rc)
PYEOF
if [[ $? -eq 0 ]]; then ok "Chequeos de integridad física"; else bad "Chequeos de integridad física"; fi

# ---------------------------- Métricas de navegación -------------------------
hdr "Métricas reportadas por los algoritmos"
echo "--- Paso 5 (VO / VO+TRN) — esperado README: VO ~430 m RMSE, VO+TRN ~93 m RMSE ---"
grep -aiE "rmse|cep" logs/paso_5.log || warn "No se encontraron líneas RMSE/CEP en logs/paso_5.log"
echo
echo "--- Paso 6 (INS / EKF) — esperado README: INS ~530 m, EKF ~104 m, CEP-50 ~78 m, CEP-95 ~192 m ---"
grep -aiE "rmse|cep" logs/paso_6.log || warn "No se encontraron líneas RMSE/CEP en logs/paso_6.log"

echo
echo "Comprobación de NaN en covarianzas del EKF:"
"$PYTHON" - <<'PYEOF'
import pandas as pd
df = pd.read_csv("data/fused_navigation_log.csv")
n = int(df.isnull().sum().sum())
print(f"Filas: {len(df)} | NaN totales: {n}")
raise SystemExit(0 if n == 0 else 1)
PYEOF
if [[ $? -eq 0 ]]; then ok "EKF sin NaN"; else bad "EKF contiene NaN (subir Q_diag)"; fi

# ---------------------------- Fin --------------------------------------------
print_summary
(( FAIL == 0 )) && echo "${G}Pipeline completado sin errores.${N}" || { echo "${R}Pipeline con errores.${N}"; exit 1; }
