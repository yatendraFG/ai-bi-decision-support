@echo off
REM ============================================================
REM  AI BI System - one-click environment setup (Windows)
REM  Creates a conda env "ai-bi" (Python 3.11), installs requirements,
REM  registers a Jupyter kernel and runs a quick check.
REM  Log: setup_log.txt
REM ============================================================
cd /d "%~dp0"
set LOG=%~dp0setup_log.txt
echo Setup started %date% %time% > "%LOG%"

set CONDA=%USERPROFILE%\anaconda3\condabin\conda.bat
set ENVPY=%USERPROFILE%\anaconda3\envs\ai-bi\python.exe

if not exist "%CONDA%" (
    echo [ERROR] Anaconda not found at %USERPROFILE%\anaconda3 >> "%LOG%"
    echo [ERROR] Anaconda not found at %USERPROFILE%\anaconda3
    goto :end
)

echo [1/5] Creating conda environment "ai-bi" with Python 3.11 ...
echo [1/5] create env >> "%LOG%"
if exist "%ENVPY%" (
    echo       already exists, skipping
    echo       env already exists >> "%LOG%"
) else (
    call "%CONDA%" create -n ai-bi python=3.11 -y >> "%LOG%" 2>&1
)
if not exist "%ENVPY%" (
    echo [ERROR] environment was not created - see setup_log.txt
    echo [ERROR] env not created >> "%LOG%"
    goto :end
)

echo [2/5] Upgrading pip ...
echo [2/5] pip upgrade >> "%LOG%"
"%ENVPY%" -m pip install --upgrade pip >> "%LOG%" 2>&1

echo [3/5] Installing requirements (TensorFlow is big, 5-10 min) ...
echo [3/5] pip install requirements >> "%LOG%"
"%ENVPY%" -m pip install -r requirements.txt ipykernel >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [ERROR] pip install failed - see setup_log.txt
    echo [ERROR] pip install failed >> "%LOG%"
    goto :end
)

echo [4/5] Registering Jupyter kernel "Python (ai-bi)" ...
echo [4/5] kernel >> "%LOG%"
"%ENVPY%" -m ipykernel install --user --name ai-bi --display-name "Python (ai-bi)" >> "%LOG%" 2>&1

echo [5/5] Checking libraries ...
echo [5/5] check >> "%LOG%"
"%ENVPY%" -c "import sys, pandas, sklearn, tensorflow as tf, matplotlib; print('OK python', sys.version.split()[0], '| pandas', pandas.__version__, '| sklearn', sklearn.__version__, '| tensorflow', tf.__version__)" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [ERROR] library check failed - see setup_log.txt
    echo [ERROR] check failed >> "%LOG%"
    goto :end
)
echo SETUP COMPLETE >> "%LOG%"
echo.
echo ===== SETUP COMPLETE =====
echo Run the project:   run_project.bat
echo Notebooks kernel:  Python (ai-bi)

:end
echo Setup finished %date% %time% >> "%LOG%"
echo.
pause
