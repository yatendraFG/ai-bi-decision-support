@echo off
REM Opens Jupyter in the notebooks folder using the "ai-bi" environment
cd /d "%~dp0"
set ENVDIR=%USERPROFILE%\anaconda3\envs\ai-bi
if not exist "%ENVDIR%\python.exe" (
    echo Environment not found. Run setup_env.bat first.
    pause
    exit /b 1
)
"%ENVDIR%\python.exe" -m notebook --notebook-dir="%~dp0notebooks"
