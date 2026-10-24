@echo off
REM Opens the AI BI dashboard in your browser (http://localhost:8501). Close this window to stop it.
cd /d "%~dp0"
set ENVPY=%USERPROFILE%\anaconda3\envs\ai-bi\python.exe
if not exist "%ENVPY%" (
    echo Environment not found. Run setup_env.bat first.
    pause
    exit /b 1
)
set TF_CPP_MIN_LOG_LEVEL=2
echo Starting dashboard at http://localhost:8501 ... (close this window to stop)
"%ENVPY%" -m streamlit run app\dashboard.py --server.port 8501 --browser.gatherUsageStats false
pause
