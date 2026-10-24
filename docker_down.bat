@echo off
REM Stops and removes the dashboard + API containers (the image stays, so the next start is fast)
cd /d "%~dp0"
docker compose down
echo Stopped.
pause
