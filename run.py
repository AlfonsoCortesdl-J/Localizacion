#!/usr/bin/env python3
"""Arranque portable del panel de navegacion aerea.

Este script usa solo la biblioteca estandar de Python. La primera vez:
  1. crea el entorno virtual .venv
  2. instala requirements.txt
  3. arranca src/realtime_hub.py
  4. abre el navegador en http://localhost:8080

Uso:
  python run.py                     # mision campus (Grob 120TP)
  python run.py --source sim        # Cessna 172
  python run.py --source live       # telemetria real / ESP32
  python run.py --no-browser        # no abrir el navegador
  python run.py --setup-only        # solo instalar dependencias
  python run.py --http-port 8090    # cambiar el puerto

Cualquier argumento que no sea --no-browser o --setup-only se pasa al hub.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import threading
import time
import venv
import webbrowser
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
STAMP = VENV / ".requirements.sha256"
HUB = ROOT / "src" / "realtime_hub.py"
MIN_PYTHON = (3, 11)


def log(message: str) -> None:
    print(f"[inicio] {message}", flush=True)


def fail(message: str, code: int = 1) -> NoReturn:
    print(f"\n[ERROR] {message}\n", flush=True)
    raise SystemExit(code)


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def requirements_digest() -> str:
    if not REQUIREMENTS.exists():
        return "sin-requirements"
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def ensure_venv() -> None:
    python = venv_python()
    if python.exists():
        return
    log("creando entorno virtual .venv (solo la primera vez)...")
    try:
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV)
    except Exception as exc:  # noqa: BLE001
        fail(
            "No se pudo crear el entorno virtual.\n"
            f"  Detalle: {exc}\n"
            "  En Linux puede faltar python3-venv: sudo apt install python3-venv"
        )
    if not python.exists():
        fail("El entorno virtual se creo pero no se encontro su interprete.")


def run_pip(args: list[str], quiet: bool = False) -> None:
    cmd = [str(venv_python()), "-m", "pip", *args]
    kwargs = {"cwd": str(ROOT)}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
    subprocess.check_call(cmd, **kwargs)


def ensure_dependencies() -> None:
    digest = requirements_digest()
    if STAMP.exists() and STAMP.read_text(encoding="utf-8").strip() == digest:
        return
    if not REQUIREMENTS.exists():
        fail(f"No existe {REQUIREMENTS.name}; no se pueden instalar dependencias.")
    log("instalando dependencias (puede tardar unos minutos la primera vez)...")
    try:
        run_pip(["install", "--upgrade", "pip"], quiet=True)
        run_pip(["install", "-r", str(REQUIREMENTS)])
    except subprocess.CalledProcessError:
        fail(
            "Fallo la instalacion de dependencias.\n"
            "  Revisa que tengas internet y Python 3.11 o superior.\n"
            "  Puedes reintentar con: python run.py --setup-only"
        )
    STAMP.write_text(digest, encoding="utf-8")
    log("dependencias listas.")


def option_value(args: list[str], name: str, default: str) -> str:
    for index, item in enumerate(args):
        if item == name and index + 1 < len(args):
            return args[index + 1]
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return default


def has_option(args: list[str], name: str) -> bool:
    return any(item == name or item.startswith(name + "=") for item in args)


def open_browser_later(url: str) -> None:
    def _open() -> None:
        time.sleep(3.0)
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_open, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Arranque portable del panel de navegacion aerea.",
        add_help=True,
    )
    parser.add_argument("--no-browser", action="store_true", help="no abrir el navegador")
    parser.add_argument("--setup-only", action="store_true", help="solo instalar dependencias")
    known, forwarded = parser.parse_known_args()

    if sys.version_info < MIN_PYTHON:
        fail(
            f"Se necesita Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} o superior "
            f"(detectado {sys.version.split()[0]}).\n"
            "  Descargalo en https://www.python.org/downloads/"
        )
    if not HUB.exists():
        fail(f"No se encontro {HUB}.")

    ensure_venv()
    ensure_dependencies()

    if known.setup_only:
        log("instalacion completa. Ya puedes ejecutar iniciar.bat / iniciar.sh.")
        return

    hub_args = list(forwarded)
    if not has_option(hub_args, "--source") and not has_option(hub_args, "--mission"):
        hub_args = ["--source", "campus", *hub_args]

    port = option_value(hub_args, "--http-port", "8080")
    url = f"http://localhost:{port}"
    if not known.no_browser:
        log(f"abriendo el panel en {url}")
        open_browser_later(url)

    log(f"iniciando hub: {url}  (Ctrl+C para detener)")
    cmd = [str(venv_python()), str(HUB), *hub_args]
    try:
        code = subprocess.call(cmd, cwd=str(ROOT))
    except KeyboardInterrupt:
        code = 0
    raise SystemExit(code)


if __name__ == "__main__":
    main()
