@echo off
setlocal
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)

if not defined PY (
  echo.
  echo No se encontro Python. Instala Python 3.11 o superior:
  echo   https://www.python.org/downloads/windows/
  echo.
  pause
  exit /b 1
)

%PY% "%~dp0empaquetar.py" %*
echo.
pause
