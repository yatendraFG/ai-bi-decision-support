@echo off
REM Starts the REST API. Interactive docs: http://127.0.0.1:8000/docs  (close this window to stop)
cd /d "%~dp0"
set ENVPY=%USERPROFILE%\anaconda3\envs\ai-bi\python.exe
if not exist "%ENVPY%" (
    echo Environment not found. Run setup_env.bat first.
    pause
    exit /b 1
)
set TF_CPP_MIN_LOG_LEVEL=2
echo API docs: http://127.0.0.1:8000/docs   (close this window to stop)
start "" http://127.0.0.1:8000/docs
"%ENVPY%" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
pause
