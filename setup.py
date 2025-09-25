# Programming Helper Agent Setup Script
# Run this script to set up the development environment

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
        else:
            print(f"❌ {description} failed")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {str(e)}")
        return False
    return True

def main():
    """Main setup function"""
    print("🚀 Programming Helper Agent Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")
    
    # Create virtual environment
    venv_path = Path("venv")
    if not venv_path.exists():
        if not run_command("python -m venv venv", "Creating virtual environment"):
            sys.exit(1)
    else:
        print("✅ Virtual environment already exists")
    
    # Determine activation command based on OS
    if sys.platform == "win32":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    print(f"\n📋 To activate virtual environment, run:")
    print(f"   {activate_cmd}")
    
    # Install requirements
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        print("⚠️  Pip upgrade failed, continuing anyway...")
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing requirements"):
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Create .env file if it doesn't exist
    env_file = Path("config/.env")
    env_example = Path("config/.env.example")
    
    if not env_file.exists() and env_example.exists():
        try:
            import shutil
            shutil.copy(env_example, env_file)
            print("✅ Created .env file from template")
            print("⚠️  Please edit config/.env and add your API keys")
        except Exception as e:
            print(f"❌ Failed to create .env file: {str(e)}")
    
    # Create data directory
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir(exist_ok=True)
        print("✅ Created data directory")
        print("📁 Add your documentation files to the data/ directory")
    
    # Create embeddings directory
    embeddings_dir = Path("embeddings")
    if not embeddings_dir.exists():
        embeddings_dir.mkdir(exist_ok=True)
        print("✅ Created embeddings directory")
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Activate virtual environment:")
    print(f"   {activate_cmd}")
    print("2. Edit config/.env and add your API keys")
    print("3. Add documentation files to data/ directory")
    print("4. Run: python app.py --initialize")
    print("5. Start using: python app.py")

if __name__ == "__main__":
    main()