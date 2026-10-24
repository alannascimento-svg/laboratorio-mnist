@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Execute primeiro: instalar_windows.bat
    exit /b 1
)
".venv\Scripts\python.exe" instalar.py --verificar
if errorlevel 1 (
    echo Confira o ambiente executando: instalar_windows.bat
    exit /b 1
)
".venv\Scripts\python.exe" app.py %*
exit /b %errorlevel%
