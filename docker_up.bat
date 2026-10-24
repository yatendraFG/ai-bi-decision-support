@echo off
REM Builds and starts the dashboard + API in Docker.
REM Dashboard: http://localhost:8080    API docs: http://localhost:8000/docs
cd /d "%~dp0"
set LOG=%~dp0docker_log.txt
echo Docker start %date% %time% > "%LOG%"

docker info >nul 2>&1
if errorlevel 1 (
    echo Starting Docker Desktop ... (first start can take 1-2 minutes)
    start "" "%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
    for /l %%i in (1,1,60) do (
        timeout /t 5 /nobreak >nul
        docker info >nul 2>&1 && goto :ready
    )
    echo [ERROR] Docker Desktop did not start. Open it manually and run this file again.
    echo [ERROR] docker not ready >> "%LOG%"
    pause
    exit /b 1
)
:ready
echo Docker is running >> "%LOG%"
echo Building image and starting containers (first build: 3-8 min) ...
docker compose up --build -d >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [ERROR] docker compose failed - see docker_log.txt
    echo [ERROR] compose failed >> "%LOG%"
    pause
    exit /b 1
)
docker compose ps >> "%LOG%" 2>&1
echo DOCKER UP >> "%LOG%"
echo.
echo ===== RUNNING IN DOCKER =====
echo Dashboard: http://localhost:8080
echo API docs:  http://localhost:8000/docs
echo Stop with docker_down.bat
timeout /t 8 /nobreak >nul
start "" http://localhost:8080
pause
