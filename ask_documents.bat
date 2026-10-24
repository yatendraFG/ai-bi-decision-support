@echo off
REM Ask questions about the company PDFs in data\docs (English or Hinglish)
cd /d "%~dp0"
set ENVPY=%USERPROFILE%\anaconda3\envs\ai-bi\python.exe
if not exist "%ENVPY%" (
    echo Environment not found. Run setup_env.bat first.
    pause
    exit /b 1
)
"%ENVPY%" -m src.rag.ask
pause
