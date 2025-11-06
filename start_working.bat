@echo off
echo ========================================
echo  SDG Synergy Mapper v2 - WORKING VERSION
echo ========================================
echo.

cd /d "E:\sdg-synergy-mapper\v2"
echo Current directory: %CD%
echo.

echo Activating virtual environment...
call "..\sdg\Scripts\activate.bat"
echo.

echo Testing imports...
python -c "import sys; sys.path.append('.'); from config.settings import SDG_INDICATORS; print('Config loaded:', len(SDG_INDICATORS), 'indicators')"
echo.

echo Starting SDG Synergy Mapper v2...
echo The application will open at: http://localhost:8501
echo Press Ctrl+C to stop the application
echo ========================================
echo.

streamlit run app/main.py --server.port 8501

echo.
echo Application stopped.
pause

