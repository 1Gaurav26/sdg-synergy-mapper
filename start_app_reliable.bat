@echo off
echo ========================================
echo  SDG Synergy Mapper v2 - Startup Script
echo ========================================
echo.

cd /d "E:\sdg-synergy-mapper\v2"
echo Current directory: %CD%
echo.

echo Activating virtual environment...
call "..\sdg\Scripts\activate.bat"
echo.

echo Checking Python environment...
python -c "import sys; print('Python executable:', sys.executable)"
echo.

echo Checking Streamlit installation...
python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"
echo.

echo Starting SDG Synergy Mapper v2...
echo.
echo The application will open in your browser at:
echo   http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo ========================================
echo.

..\sdg\Scripts\streamlit.exe run app/main.py --server.port 8501

echo.
echo Application stopped.
pause

