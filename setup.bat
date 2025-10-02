@echo off
REM Quick setup script for QR Code Detection System (Windows)

echo 🚀 Setting up QR Code Detection System...

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo 📁 Creating directories...
if not exist "dataset\train_images" mkdir dataset\train_images
if not exist "dataset\test_images" mkdir dataset\test_images
if not exist "processed_data" mkdir processed_data
if not exist "runs" mkdir runs

echo ✅ Setup complete!
echo.
echo 📝 Next steps:
echo    1. Place your training images in dataset/train_images/
echo    2. Create annotations.json (or use annotate.py)
echo    3. Run: python train.py
echo    4. Run: python infer.py --input demo_images/ --output submission.json
echo.
echo 💡 Activate the virtual environment with: venv\Scripts\activate
pause

