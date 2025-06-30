@echo off
echo Starting TrackMyPDB Streamlit Application...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

REM Check if Streamlit is installed
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Launch the application
echo Launching TrackMyPDB...
echo Application will open in your default browser
echo Press Ctrl+C to stop the application
echo.

streamlit run streamlit_app.py

pause 