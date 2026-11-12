# ESP32-CAM, telemetría y mapa real

## Flujo implementado

```text
ESP32-CAM hacia abajo --Wi-Fi/MJPEG--> PC o Jetson
STM32 + IMU ------------UART/USB-----> PC o Jetson
PC/Jetson: captura JPEG + hora + última muestra IMU --> data/live_capture/
```

La ESP32-CAM transmite JPEG/MJPEG por Wi-Fi. No se manda el video por el puerto
serial. El puerto serial queda para el STM32 y sus datos de IMU.

## Capturar video

Primero abre la URL MJPEG en un navegador para confirmar la dirección que anuncia
el firmware. La ruta común del ejemplo CameraWebServer es `http://IP:81/stream`,
pero usa la ruta/IP que realmente muestre tu ESP32-CAM.

Desde la carpeta raíz del proyecto:

```powershell
py src\live_camera_capture.py --stream-url http://192.168.4.1:81/stream
```

Para capturar también la IMU por serial:

```powershell
py src\live_camera_capture.py --stream-url http://192.168.4.1:81/stream --port COM3 --baud 115200
```

Pulsa `q` para terminar la ventana de captura o `Ctrl+C` en modo `--headless`.
Las imágenes quedan en `data/live_capture/frames/`; el manifiesto sincronizado
queda en `data/live_capture/frames.csv`. `--save-every 5` guarda uno de cada cinco
fotogramas. En Jetson sin pantalla gráfica usa `--headless`.

El receptor acepta un ONNX opcional:

```powershell
py src\live_camera_capture.py --stream-url http://192.168.4.1:81/stream --onnx-model data\models\terrain_unet.onnx
```

Esto requiere un modelo de segmentación ya entrenado y exportado con entrada RGB
normalizada como ImageNet y salida de logits por clase. Sin `--onnx-model` solo
captura y registra; **aún no segmenta**.

## Telemetría y sincronización

Cada fila de `frames.csv` contiene el nombre/ruta de imagen, hora UTC, reloj
monotónico de la computadora, tiempo relativo, edad de la última muestra IMU y
acelerómetro/giroscopio. El lector también acepta estos campos JSON opcionales
del STM32: `lat`/`latitude_deg`, `lon`/`longitude_deg`, `alt_agl_m`/
`altitude_agl_m`, `roll_deg`, `pitch_deg`, `yaw_deg`. Los campos de posición y
actitud quedarán vacíos mientras el firmware no los transmita. El firmware
MPU6050 actual solo manda tiempo, aceleración y giro; una IMU por sí sola no da
latitud/longitud GPS ni altitud AGL absoluta.

## Preparar el nuevo mapa

Se necesita un ortomosaico georreferenciado de la zona de vuelo. Puede venir de
un levantamiento propio o de una fuente cartográfica apropiada. Exporta/guarda
el GeoTIFF en orientación norte-arriba y en una proyección métrica (por ejemplo,
UTM). La herramienta genera los PNG y metadatos esperados por `MapMatcher`:

```powershell
py -m pip install rasterio pyproj
py src\prepare_reference_map.py --geotiff maps\ortomosaico.tif --output-dir data\real_map
```

Para asociar una máscara de clases, proporciona un GeoTIFF de una banda que tenga
exactamente el mismo tamaño, CRS y transform que el ortomosaico:

```powershell
py src\prepare_reference_map.py --geotiff maps\ortomosaico.tif --labels maps\mascara_clases.tif --output-dir data\real_map
```

Si tienes un JSON que asocie nombres de clases con IDs, inclúyelo para conservar
esa leyenda en los metadatos: `--class-map maps\clases.json`.

Sin `--labels`, el resultado sirve para matching visual ORB, pero no para entrenar
la segmentación. Las máscaras sintéticas actuales no son etiquetas del terreno
real. `visual_navigation.py` permite seleccionar el mapa con `--map` y sus
metadatos con `--map-meta`; sus fotogramas CSV también deben incluir latitud,
longitud, altitud AGL, actitud y nombres de imagen válidos para poder estimar y
evaluar una trayectoria real.
