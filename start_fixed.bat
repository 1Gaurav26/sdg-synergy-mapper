@echo off
echo ========================================
echo  SDG Synergy Mapper v2 - BUG FIXED!
echo ========================================
echo.

cd /d "E:\sdg-synergy-mapper\v2"
echo Current directory: %CD%
echo.

echo Activating virtual environment...
call "..\sdg\Scripts\activate.bat"
echo.

echo Starting SDG Synergy Mapper v2 with bug fixes...
echo The ML Models section should now work properly!
echo.
echo The application will open at: http://localhost:8501
echo Press Ctrl+C to stop the application
echo ========================================
echo.

streamlit run app/main.py --server.port 8501

echo.
echo Application stopped.
pause

