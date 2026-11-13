# Localización · Navegación aérea simulada y real

Panel de navegación aérea con mapa 3D del campus **TecMM / TSJ Zapopan**, aviones
y dron simulados, cámaras en vivo, telemetría en tiempo real y fusión de sensores
(EKF + INS) con vista de posición estimada.

## Arranque rápido

Solo necesitas **Python 3.11 o superior** instalado. La primera vez el lanzador
crea el entorno virtual, instala las dependencias y abre el navegador.

### Windows

Doble clic en **`iniciar.bat`**.

### Linux / macOS

```bash
./iniciar.sh
```

Si el archivo no tiene permisos de ejecución:

```bash
chmod +x iniciar.sh
```

### Manual (cualquier sistema)

```bash
python run.py
```

El panel queda en **http://localhost:8080**.

## Qué hace el arranque

`run.py` se encarga de todo:

1. Crea `.venv/` si no existe.
2. Instala `requirements.txt` si cambió (la primera vez tarda unos minutos).
3. Arranca `src/realtime_hub.py`.
4. Abre el navegador cuando el servidor está listo.

Opciones útiles:

```bash
python run.py --setup-only            # solo instalar dependencias
python run.py --no-browser            # no abrir el navegador
python run.py --http-port 8090        # otro puerto (WS 8091, UDP 9877)
python run.py --source sim            # misión Cessna 172
python run.py --source campus         # misión Grob 120TP (por defecto)
python run.py --source live           # telemetría real / ESP32
python run.py --host 0.0.0.0          # accesible desde la red local
```

Todos los argumentos que no sean `--no-browser` o `--setup-only` se pasan tal cual
al hub, así que también puedes usar `--camera`, `--camera2`, `--port`, `--baud`,
etc.

## Misiones

| Misión | Fuente | Qué muestra |
| --- | --- | --- |
| **Cessna 172** | `--source sim` | Vuelo JSBSim con entorno original, corredor GPS-denied y cámaras simuladas. |
| **Grob 120TP · TSJ** | `--source campus` | Inspección del campus entre edificios, ruta editable y cámaras nadir/cielo. |
| **Telemetría real** | `--source live` o `--source serial` | Cámaras ESP32-CAM y dron en la posición estimada. |

En la misión del campus puedes pulsar **Editar ruta**, hacer clic sobre el mapa 3D
para colocar waypoints y **Aplicar** para recalcular el recorrido en vivo.

## Cámaras reales

- **Webcam local:** `python run.py --source live --camera 0 --camera2 1`
- **ESP32-CAM (MJPEG):** `python run.py --source live --camera http://192.168.4.1:81/stream`
- También puedes pegar la URL desde el panel, en **Cámaras reales**.

Sin cámara conectada el panel muestra una imagen sintética de respaldo; no se
rompe nada.

## Puertos

| Servicio | Puerto |
| --- | --- |
| Panel HTTP / API | 8080 |
| WebSocket | 8081 |
| Telemetría UDP | 9876 |

Si un puerto está ocupado usa `--http-port 8090` (y ajusta los demás si hace
falta: `--ws-port`, `--udp-port`).

## Llevarlo a otra PC

### Con Git

```bash
git clone https://github.com/AlfonsoCortesdl-J/Localizacion.git
cd Localizacion
```

Luego ejecuta `iniciar.bat` o `iniciar.sh`. Las dependencias se instalan solas.

### Con un ZIP portable

En la PC de origen:

```bash
python empaquetar.py                 # paquete completo
python empaquetar.py --sin-camaras   # sin fotogramas simulados (~511 MB menos)
```

Se genera `Localizacion_portable.zip`. En la otra PC: descomprime y ejecuta
`iniciar.bat` (Windows) o `iniciar.sh` (Linux/macOS).

> Sin `data/camera_ground` y `data/camera_sky` el hub usa cámaras sintéticas de
> respaldo, pero todo lo demás (mapa, misiones, telemetría, ruta, PFD) funciona
> igual.

## Estructura

```text
run.py                 arranque portable
iniciar.bat/.sh        lanzadores de un clic
empaquetar.py          genera el ZIP para otra PC
src/realtime_hub.py    servidor HTTP/WS/UDP, fuentes y cámaras
src/campus_scene.py    escena y misión del campus
src/web/dashboard.html panel principal
data/                  misiones, mapas, modelos y registros
docs/                  documentación de hardware y captura
```

## Problemas comunes

- **“No se encontró Python”**: instala Python 3.11+ y marca
  *Add python.exe to PATH*. En Windows también sirve el lanzador `py`.
- **Falla la instalación de dependencias**: revisa la conexión a internet y
  reintenta con `python run.py --setup-only`.
- **El puerto 8080 está ocupado**: `python run.py --http-port 8090`.
- **No veo las cámaras**: confirma la URL MJPEG en el navegador; en modo campus
  las cámaras reales están ocultas a propósito.
- **Cambios en el panel no aparecen**: recarga con `Ctrl+F5` para saltar la caché.
