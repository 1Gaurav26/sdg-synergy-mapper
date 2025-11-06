# SDG Synergy Mapper v2 - Troubleshooting Guide

## 🚨 Common Issues and Solutions

### Issue 1: "streamlit: The term 'streamlit' is not recognized"

**Problem**: Virtual environment not properly activated in PowerShell

**Solutions**:

#### Method 1: Use PowerShell Script (Recommended)
```powershell
# Navigate to v2 directory
cd E:\sdg-synergy-mapper\v2

# Run PowerShell script
.\start_app.ps1
```

#### Method 2: Manual PowerShell Activation
```powershell
# Navigate to v2 directory
cd E:\sdg-synergy-mapper\v2

# Activate virtual environment
& "..\sdg\Scripts\Activate.ps1"

# Verify activation (should show sdg environment path)
python -c "import sys; print(sys.executable)"

# Run streamlit
streamlit run app/main.py
```

#### Method 3: Use Command Prompt Instead
```cmd
# Open Command Prompt (cmd.exe)
cd E:\sdg-synergy-mapper\v2
..\sdg\Scripts\activate.bat
streamlit run app/main.py
```

#### Method 4: Direct Path Execution
```powershell
# Use full path to streamlit
E:\sdg-synergy-mapper\sdg\Scripts\streamlit.exe run E:\sdg-synergy-mapper\v2\app\main.py
```

### Issue 2: PowerShell Execution Policy Error

**Problem**: PowerShell blocks script execution

**Solution**:
```powershell
# Set execution policy for current user
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or run script with bypass
PowerShell -ExecutionPolicy Bypass -File start_app.ps1
```

### Issue 3: Port Already in Use

**Problem**: Port 8501 is already occupied

**Solutions**:
```powershell
# Use different port
streamlit run app/main.py --server.port 8502

# Or kill process using port 8501
netstat -ano | findstr :8501
taskkill /PID <PID_NUMBER> /F
```

### Issue 4: Import Errors

**Problem**: ModuleNotFoundError when running the app

**Solutions**:
```powershell
# Verify virtual environment is active
python -c "import sys; print(sys.executable)"
# Should show: E:\sdg-synergy-mapper\sdg\Scripts\python.exe

# Reinstall requirements
pip install -r requirements.txt

# Test imports
python -c "import streamlit, pandas, plotly; print('All imports successful')"
```

### Issue 5: Application Won't Start

**Problem**: Application crashes on startup

**Solutions**:
```powershell
# Check for syntax errors
python -m py_compile app/main.py

# Run with verbose output
streamlit run app/main.py --logger.level debug

# Check Python version (should be 3.8+)
python --version
```

## 🔧 Quick Fixes

### Reset Virtual Environment
```powershell
# Remove and recreate virtual environment
cd E:\sdg-synergy-mapper
Remove-Item -Recurse -Force sdg
python -m venv sdg
cd v2
& "..\sdg\Scripts\Activate.ps1"
pip install -r requirements.txt
```

### Verify Installation
```powershell
# Check all key packages
python -c "
import streamlit; print('Streamlit:', streamlit.__version__)
import pandas; print('Pandas:', pandas.__version__)
import plotly; print('Plotly:', plotly.__version__)
import sklearn; print('Scikit-learn:', sklearn.__version__)
print('All packages installed successfully!')
"
```

### Test Application Components
```powershell
# Test configuration loading
python -c "
import sys; sys.path.append('.')
from config.settings import SDG_INDICATORS
print('SDG Indicators loaded:', len(SDG_INDICATORS))
"

# Test data processing
python -c "
import sys; sys.path.append('.')
from utils.data_processing import DataValidator
print('Data processing module loaded successfully')
"
```

## 📋 Step-by-Step Startup Guide

### For PowerShell Users:
1. Open PowerShell
2. Navigate to project: `cd E:\sdg-synergy-mapper\v2`
3. Run: `.\start_app.ps1`

### For Command Prompt Users:
1. Open Command Prompt
2. Navigate to project: `cd E:\sdg-synergy-mapper\v2`
3. Run: `start_app.bat`

### For Manual Startup:
1. Navigate to: `E:\sdg-synergy-mapper\v2`
2. Activate environment: `..\sdg\Scripts\activate.bat` (cmd) or `& "..\sdg\Scripts\Activate.ps1"` (PowerShell)
3. Run: `streamlit run app/main.py`

## 🌐 Access the Application

Once successfully started, open your browser and go to:
- **http://localhost:8501** (default)
- **http://localhost:8502** (if port 8501 is busy)

## 📞 Still Having Issues?

If you're still experiencing problems:

1. **Check Python Version**: Must be 3.8 or higher
2. **Verify Virtual Environment**: Should show `(sdg)` in prompt
3. **Check Dependencies**: Run `pip list` to see installed packages
4. **Try Different Terminal**: Use Command Prompt instead of PowerShell
5. **Check File Paths**: Ensure you're in the correct directory

## ✅ Success Indicators

You'll know everything is working when you see:
- Virtual environment activated (shows `(sdg)` in prompt)
- Streamlit starts without errors
- Browser opens automatically to the application
- Application loads with the SDG Synergy Mapper v2 interface

**Happy analyzing! 🌐✨**

