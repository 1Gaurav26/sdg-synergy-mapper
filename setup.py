"""
Setup script for SDG Synergy Mapper v2
Automated installation and configuration
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "cache",
        "logs",
        "exports",
        "models/saved",
        "static/images",
        "templates/reports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_env_file():
    """Create .env file with default settings"""
    print("⚙️ Creating configuration file...")
    
    env_content = """# SDG Synergy Mapper v2 Configuration

# API Keys (optional - leave empty to use public APIs)
WORLD_BANK_API_KEY=
UN_STATS_API_KEY=
OECD_API_KEY=

# Database Configuration
DATABASE_URL=sqlite:///users.db

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here

# Cache Settings
CACHE_TTL=3600
MAX_CACHE_SIZE=1000

# Application Settings
DEBUG=True
LOG_LEVEL=INFO

# Export Settings
MAX_EXPORT_SIZE=100
EXPORT_FORMATS=pdf,html,excel,csv,json,png,svg
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        env_file.write_text(env_content)
        print("✅ Created .env file")
    else:
        print("ℹ️ .env file already exists")

def create_sample_config():
    """Create sample configuration files"""
    print("📋 Creating sample configurations...")
    
    # Create sample data config
    sample_data_config = {
        "countries": ["USA", "China", "India", "Brazil", "Germany", "Japan"],
        "indicators": ["gdp_per_capita", "life_expectancy", "poverty_rate"],
        "start_year": 2015,
        "end_year": 2023,
        "data_source": "sample"
    }
    
    import json
    config_file = Path("config/sample_config.json")
    config_file.parent.mkdir(exist_ok=True)
    config_file.write_text(json.dumps(sample_data_config, indent=2))
    print("✅ Created sample configuration")

def run_tests():
    """Run basic tests to verify installation"""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        import streamlit
        import pandas
        import numpy
        import plotly
        import sklearn
        print("✅ All required packages imported successfully")
        
        # Test configuration loading
        sys.path.append(str(Path(__file__).parent))
        from config.settings import SDG_INDICATORS
        print(f"✅ Configuration loaded: {len(SDG_INDICATORS)} SDG indicators")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def create_startup_script():
    """Create startup script"""
    print("🚀 Creating startup script...")
    
    if platform.system() == "Windows":
        startup_script = """@echo off
echo Starting SDG Synergy Mapper v2...
cd /d "%~dp0"
streamlit run app/main.py --server.port 8501 --server.headless true
pause
"""
        startup_file = Path("start_app.bat")
    else:
        startup_script = """#!/bin/bash
echo "Starting SDG Synergy Mapper v2..."
cd "$(dirname "$0")"
streamlit run app/main.py --server.port 8501 --server.headless true
"""
        startup_file = Path("start_app.sh")
    
    startup_file.write_text(startup_script)
    
    if platform.system() != "Windows":
        os.chmod(startup_file, 0o755)
    
    print(f"✅ Created startup script: {startup_file}")

def main():
    """Main setup function"""
    print("🌐 SDG Synergy Mapper v2 Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create configuration files
    create_env_file()
    create_sample_config()
    
    # Run tests
    if not run_tests():
        print("❌ Setup failed during testing")
        sys.exit(1)
    
    # Create startup script
    create_startup_script()
    
    print("\n" + "=" * 40)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python setup.py start")
    print("2. Or run: streamlit run app/main.py")
    print("3. Open your browser to: http://localhost:8501")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "start":
        # Start the application
        print("🚀 Starting SDG Synergy Mapper v2...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/main.py"])
    else:
        # Run setup
        main()

