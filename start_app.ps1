# PowerShell script to start SDG Synergy Mapper v2
Write-Host "Starting SDG Synergy Mapper v2..." -ForegroundColor Green
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "app\main.py")) {
    Write-Host "Error: Please run this script from the v2 directory" -ForegroundColor Red
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
    Write-Host "Expected: E:\sdg-synergy-mapper\v2" -ForegroundColor Yellow
    pause
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
try {
    & "..\sdg\Scripts\Activate.ps1"
    Write-Host "Virtual environment activated successfully!" -ForegroundColor Green
} catch {
    Write-Host "Error activating virtual environment: $_" -ForegroundColor Red
    pause
    exit 1
}

# Check if streamlit is available
Write-Host "Checking Streamlit installation..." -ForegroundColor Yellow
try {
    $streamlitVersion = python -c "import streamlit; print(streamlit.__version__)"
    Write-Host "Streamlit version: $streamlitVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Streamlit not found in virtual environment" -ForegroundColor Red
    Write-Host "Please run: pip install -r requirements.txt" -ForegroundColor Yellow
    pause
    exit 1
}

# Start the application
Write-Host ""
Write-Host "Starting Streamlit application..." -ForegroundColor Green
Write-Host "The application will open in your default browser at: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Yellow
Write-Host ""

try {
    streamlit run app/main.py --server.port 8501
} catch {
    Write-Host "Error starting application: $_" -ForegroundColor Red
    pause
}

