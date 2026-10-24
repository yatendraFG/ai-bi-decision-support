@echo off
REM AI BI agent: ask business questions in English or Hinglish; reports are saved in reports\agent
cd /d "%~dp0"
set ENVPY=%USERPROFILE%\anaconda3\envs\ai-bi\python.exe
if not exist "%ENVPY%" (
    echo Environment not found. Run setup_env.bat first.
    pause
    exit /b 1
)
set TF_CPP_MIN_LOG_LEVEL=2
"%ENVPY%" -m src.agent.ask_agent
pause
