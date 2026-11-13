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
  echo No se encontro Python.
  echo Instala Python 3.11 o superior desde:
  echo   https://www.python.org/downloads/windows/
  echo Durante la instalacion marca "Add python.exe to PATH".
  echo.
  pause
  exit /b 1
)

%PY% "%~dp0run.py" %*
if errorlevel 1 (
  echo.
  echo El panel termino con errores. Revisa el mensaje anterior.
  pause
)
