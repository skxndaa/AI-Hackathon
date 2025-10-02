#!/bin/bash
# Quick setup script for QR Code Detection System

echo "🚀 Setting up QR Code Detection System..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p dataset/train_images
mkdir -p dataset/test_images
mkdir -p processed_data
mkdir -p runs

echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Place your training images in dataset/train_images/"
echo "   2. Create annotations.json (or use annotate.py)"
echo "   3. Run: python train.py"
echo "   4. Run: python infer.py --input demo_images/ --output submission.json"
echo ""
echo "💡 Activate the virtual environment with: source venv/bin/activate"

