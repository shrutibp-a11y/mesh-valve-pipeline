@echo off
REM ============================================================
REM  One-Click Setup (Windows)
REM  Python 3.13 compatible
REM  Double-click this file to set everything up automatically
REM ============================================================

echo.
echo ====================================================
echo   Setup  (Python 3.13)
echo ====================================================
echo.

REM Check Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python from https://python.org
    echo Make sure to tick "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check Python version is 3.13+
python -c "import sys; exit(0) if sys.version_info >= (3,11) else exit(1)" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Python 3.11 or newer recommended.
    echo Your current version may work but is not tested.
    echo.
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)
echo Done.
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip first (important for Python 3.13)
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install all dependencies
echo Installing dependencies (this takes 2-5 minutes)...
echo.
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Some packages failed to install.
    echo Try running this manually:
    echo   pip install nibabel>=5.3.0 numpy>=2.0.0 scipy scikit-image>=0.26.0 scikit-learn trimesh matplotlib tqdm
    pause
    exit /b 1
)

echo.
echo ====================================================
echo   Setup COMPLETE!
echo ====================================================
echo.
echo Next steps:
echo   1. Put your ACDC data in:  data\training\patient001\
echo   2. Open terminal and run:  venv\Scripts\activate.bat
echo   3. Then run:               python run_all.py
echo.
pause
