@echo off
echo TrackMyPDB Installation Script
echo ==============================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing required packages...
pip install -r requirements.txt

REM Check RDKit installation specifically
echo.
echo Checking RDKit installation...
python -c "from rdkit import Chem; print('RDKit successfully installed!')" 2>nul
if errorlevel 1 (
    echo WARNING: RDKit installation may have failed
    echo Trying alternative installation method...
    conda install -c conda-forge rdkit -y 2>nul
    if errorlevel 1 (
        echo Please install RDKit manually:
        echo   conda install -c conda-forge rdkit
        echo   OR
        echo   pip install rdkit-pypi
    )
)

echo.
echo Installation completed!
echo To run the application, execute: run.bat
echo Or use: streamlit run streamlit_app.py
pause 