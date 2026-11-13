#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo
  echo "No se encontro Python. Instala Python 3.11 o superior:"
  echo "  https://www.python.org/downloads/"
  echo
  exit 1
fi

exec "$PY" run.py "$@"
