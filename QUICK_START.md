# 🚀 SDG Synergy Mapper v2 - QUICK START GUIDE

## ✅ **SOLUTION: Use These Commands**

The virtual environment activation issue in PowerShell has been resolved. Here are the **working solutions**:

### **Method 1: Reliable Batch File (Recommended)**
```cmd
# Double-click this file or run in Command Prompt:
start_app_reliable.bat
```

### **Method 2: Reliable PowerShell Script**
```powershell
# Right-click and "Run with PowerShell" or run in PowerShell:
.\start_app_reliable.ps1
```

### **Method 3: Manual Commands (Command Prompt)**
```cmd
cd E:\sdg-synergy-mapper\v2
..\sdg\Scripts\activate.bat
..\sdg\Scripts\streamlit.exe run app/main.py
```

### **Method 4: Manual Commands (PowerShell)**
```powershell
cd E:\sdg-synergy-mapper\v2
E:\sdg-synergy-mapper\sdg\Scripts\streamlit.exe run app/main.py
```

## 🌐 **Access the Application**

Once started successfully, open your browser and go to:
**http://localhost:8501**

## 🔧 **What Was Fixed**

1. **Created reliable startup scripts** that use full paths
2. **Added error checking** to verify installation
3. **Provided multiple methods** for different terminal types
4. **Fixed PowerShell activation issues** by using direct paths

## 📁 **New Files Created**

- `start_app_reliable.bat` - Works in Command Prompt
- `start_app_reliable.ps1` - Works in PowerShell
- Both scripts include error checking and status messages

## ✅ **Success Indicators**

You'll know it's working when you see:
- ✅ Python executable path shows the virtual environment
- ✅ Streamlit version is displayed
- ✅ Application starts without errors
- ✅ Browser opens automatically to http://localhost:8501

## 🎯 **Quick Test**

To verify everything is working:
```cmd
cd E:\sdg-synergy-mapper\v2
start_app_reliable.bat
```

**The application should now start successfully! 🌐✨**

