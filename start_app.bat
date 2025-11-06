@echo off
echo Starting SDG Synergy Mapper v2...
echo.
echo Activating virtual environment...
call ..\sdg\Scripts\activate.bat
echo.
echo Checking Python environment...
python -c "import sys; print('Python:', sys.executable)"
echo.
echo Starting Streamlit application...
echo The application will open at: http://localhost:8501
echo Press Ctrl+C to stop the application
echo.
streamlit run app/main.py --server.port 8501
pause
