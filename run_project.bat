@echo off
REM Runs the full pipeline inside the "ai-bi" environment. Output also saved to run_log.txt
cd /d "%~dp0"
set ENVPY=%USERPROFILE%\anaconda3\envs\ai-bi\python.exe
if not exist "%ENVPY%" (
    echo Environment not found. Run setup_env.bat first.
    pause
    exit /b 1
)
set TF_CPP_MIN_LOG_LEVEL=2
echo Running pipeline (Steps 2-8 + API check) ... output also in run_log.txt
"%ENVPY%" run_pipeline.py --skip-generate %* > run_log.txt 2>&1
type reports\summary.md
echo.
echo Full log: run_log.txt
pause
