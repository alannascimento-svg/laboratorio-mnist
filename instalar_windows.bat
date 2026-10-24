@echo off
setlocal
cd /d "%~dp0"
for %%V in (3.12 3.13 3.11) do (
    py -%%V -c "import sys; sys.exit(sys.version_info[:2] not in ((3,11),(3,12),(3,13)))" >nul 2>&1
    if not errorlevel 1 (
        py -%%V instalar.py %*
        if errorlevel 1 exit /b 1
        exit /b 0
    )
)
python -c "import sys; sys.exit(sys.version_info[:2] not in ((3,11),(3,12),(3,13)))" >nul 2>&1
if not errorlevel 1 (
    python instalar.py %*
    if errorlevel 1 exit /b 1
    exit /b 0
)
echo Instale Python 3.12 ^(ou 3.11/3.13^) e execute este instalador novamente.
exit /b 1
