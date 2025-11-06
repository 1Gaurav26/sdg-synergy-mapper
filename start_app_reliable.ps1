# SDG Synergy Mapper v2 - Reliable PowerShell Startup Script
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SDG Synergy Mapper v2 - Startup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Set working directory
Set-Location "E:\sdg-synergy-mapper\v2"
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Green
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "app\main.py")) {
    Write-Host "ERROR: Cannot find app\main.py" -ForegroundColor Red
    Write-Host "Please ensure you're running this from the correct directory" -ForegroundColor Yellow
    pause
    exit 1
}

# Check Python environment
Write-Host "Checking Python environment..." -ForegroundColor Yellow
$pythonPath = "E:\sdg-synergy-mapper\sdg\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    Write-Host "ERROR: Python not found at $pythonPath" -ForegroundColor Red
    Write-Host "Please check your virtual environment setup" -ForegroundColor Yellow
    pause
    exit 1
}

# Check Streamlit installation
Write-Host "Checking Streamlit installation..." -ForegroundColor Yellow
try {
    $streamlitVersion = & $pythonPath -c "import streamlit; print(streamlit.__version__)"
    Write-Host "Streamlit version: $streamlitVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Streamlit not found in virtual environment" -ForegroundColor Red
    Write-Host "Please run: pip install -r requirements.txt" -ForegroundColor Yellow
    pause
    exit 1
}

# Start the application
Write-Host ""
Write-Host "Starting SDG Synergy Mapper v2..." -ForegroundColor Green
Write-Host ""
Write-Host "The application will open in your browser at:" -ForegroundColor Cyan
Write-Host "  http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

try {
    $streamlitPath = "E:\sdg-synergy-mapper\sdg\Scripts\streamlit.exe"
    & $streamlitPath run app/main.py --server.port 8501
} catch {
    Write-Host "ERROR starting application: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Alternative: Try running from Command Prompt instead:" -ForegroundColor Yellow
    Write-Host "  start_app_reliable.bat" -ForegroundColor Yellow
    pause
}

Write-Host ""
Write-Host "Application stopped." -ForegroundColor Green
pause

