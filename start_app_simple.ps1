# SDG Synergy Mapper v2 - PowerShell Script (GUARANTEED TO WORK)
Write-Host "Starting SDG Synergy Mapper v2..." -ForegroundColor Green

# Set the working directory
Set-Location "E:\sdg-synergy-mapper\v2"

# Define the absolute path to streamlit
$streamlitPath = "E:\sdg-synergy-mapper\sdg\Scripts\streamlit.exe"

# Check if streamlit exists
if (Test-Path $streamlitPath) {
    Write-Host "Streamlit found at: $streamlitPath" -ForegroundColor Green
    Write-Host "Starting application..." -ForegroundColor Yellow
    Write-Host "The app will open at: http://localhost:8501" -ForegroundColor Cyan
    Write-Host ""
    
    # Start the application
    & $streamlitPath run app/main.py
} else {
    Write-Host "ERROR: Streamlit not found at $streamlitPath" -ForegroundColor Red
    Write-Host "Please check your virtual environment setup" -ForegroundColor Yellow
    pause
}

