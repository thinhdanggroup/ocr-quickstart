#!/bin/bash
# Installation script for macOS/Linux

echo "🚀 Installing OCR ID Card Recognition Application"
echo "=================================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if Tesseract is installed
echo ""
echo "📦 Checking for Tesseract OCR..."
if ! command -v tesseract &> /dev/null; then
    echo "⚠️  Tesseract not found. Installing..."
    
    # Detect OS and install
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            echo "Installing via Homebrew..."
            brew install tesseract
        else
            echo "❌ Homebrew not found. Please install Homebrew first:"
            echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "Installing via apt..."
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr
    else
        echo "❌ Unsupported OS. Please install Tesseract manually."
        exit 1
    fi
else
    echo "✅ Tesseract found: $(tesseract --version | head -n 1)"
fi

# Create virtual environment
echo ""
echo "📦 Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Installation complete!"
echo ""
echo "To use the application:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run: python ocr_id_card.py <image_path>"
echo ""
echo "Example:"
echo "  python ocr_id_card.py sample_id.jpg"
echo ""
echo "For help:"
echo "  python ocr_id_card.py --help"
