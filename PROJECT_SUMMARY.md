# 🎉 OCR ID Card Recognition Application

## ✅ Project Complete!

Your OCR ID Card recognition application has been successfully implemented following the comprehensive tutorial in `docs/instruction.md`.

## 📁 Project Structure

```
ocr-quickstart/
├── src/                          # Core modules
│   ├── __init__.py
│   ├── preprocessing.py          # Image preprocessing (grayscale, denoise, binarize, deskew)
│   └── ocr.py                    # OCR engine & text extraction
│
├── docs/
│   └── instruction.md            # Comprehensive OCR tutorial
│
├── ocr_id_card.py                # Main CLI application ⭐
├── examples.py                   # Usage examples
├── test_installation.py          # Installation verification
├── install.sh                    # Automated installer
│
├── README.md                     # Full documentation
├── QUICKSTART.md                 # Quick start guide
├── requirements.txt              # Python dependencies
└── .gitignore                    # Git ignore rules
```

## 🚀 Getting Started

### 1. Test Your Installation

```bash
python test_installation.py
```

This will verify:

- Python version (3.8+)
- All dependencies (OpenCV, Tesseract, pytesseract, pandas, numpy, Pillow)
- Tesseract binary installation
- Available language packs
- Basic OCR functionality

### 2. Install Dependencies (if needed)

**Automatic (Recommended):**

```bash
./install.sh
```

**Manual:**

```bash
# Install Tesseract
brew install tesseract  # macOS

# Install Python packages
pip install -r requirements.txt
```

### 3. Run OCR on an ID Card

```bash
# Basic usage
python ocr_id_card.py <your_id_card_image.jpg>

# With options
python ocr_id_card.py id_card.jpg --lang eng --conf 60 --psm 6 --output results/
```

## 🎯 Key Features Implemented

### ✨ Image Preprocessing

- Grayscale conversion with Gaussian denoising
- Otsu and adaptive thresholding
- Automatic deskewing (rotation correction)
- Noise removal and speckle cleaning
- Smart upsampling for small text

### 🔍 OCR Processing

- Tesseract LSTM engine (OEM 3)
- Configurable page segmentation modes
- Confidence-based filtering
- Multi-language support
- Word-level bounding boxes

### 📊 Output Formats

- **Annotated Images** - Bounding boxes on detected words
- **Searchable PDFs** - Invisible text layer
- **TSV Files** - Detailed OCR data with confidence scores
- **JSON** - Structured field extraction

### 🆔 ID Card Field Extraction

Automatically extracts:

- ID Number (multiple formats)
- Full Name
- Date of Birth
- Sex/Gender
- Issue Date
- Expiry Date
- Nationality
- Address

## 📖 Documentation

- **Full Guide**: `README.md` - Complete documentation with examples
- **Quick Start**: `QUICKSTART.md` - Get up and running fast
- **Tutorial**: `docs/instruction.md` - Deep dive into OCR concepts
- **Examples**: `examples.py` - Code examples for common use cases

## 🎨 Example Usage

### CLI Application

```bash
# Basic ID card processing
python ocr_id_card.py license.jpg

# Multi-language (English + Vietnamese)
python ocr_id_card.py vietnam_id.jpg --lang eng+vie

# High confidence filter
python ocr_id_card.py id.jpg --conf 75

# Save structured data
python ocr_id_card.py id.jpg --json output.json
```

### Python API

```python
from src.ocr import ocr_id_card

results = ocr_id_card(
    image_path="id_card.jpg",
    lang="eng",
    psm=6,
    min_conf=60
)

print(results['text'])         # Full text
print(results['fields'])       # Extracted fields
print(results['boxes'])        # Bounding boxes
```

## 🔧 Configuration Options

### Page Segmentation Modes (--psm)

- `6` - Uniform block of text (default, best for ID cards) ⭐
- `11` - Sparse text (good for photos)
- `3` - Fully automatic
- `7` - Single text line

### Language Support (--lang)

- `eng` - English (default)
- `eng+vie` - English + Vietnamese
- `eng+fra` - English + French
- See installed: `tesseract --list-langs`

### Confidence Threshold (--conf)

- `60` - Default (balanced)
- `70-80` - Stricter filtering
- `40-50` - More permissive (for low quality images)

## 📊 Output Files

After processing `sample_id.jpg`, you'll get:

```
output/
├── sample_id_boxes.jpg          # Annotated with bounding boxes
├── sample_id_preprocessed.jpg   # Binary processed image
├── sample_id_ocr.tsv           # Detailed OCR data
└── sample_id.pdf                # Searchable PDF
```

## 🛠 Troubleshooting

### Issue: "tesseract command not found"

**Solution**: Install Tesseract OCR

```bash
brew install tesseract  # macOS
sudo apt-get install tesseract-ocr  # Ubuntu
```

### Issue: Low accuracy

**Solutions**:

- Use higher resolution images (300+ DPI)
- Ensure good lighting
- Try `--conf 50` for lower threshold
- Try `--psm 11` for photos
- Install correct language pack

### Issue: Missing dependencies

**Solution**: Run installation test

```bash
python test_installation.py
pip install -r requirements.txt
```

## 📚 Next Steps

1. **Test the application**

   ```bash
   python test_installation.py
   ```

2. **Try the examples**

   ```bash
   # Edit examples.py and uncomment desired examples
   python examples.py
   ```

3. **Process your ID cards**

   ```bash
   python ocr_id_card.py your_id_card.jpg
   ```

4. **Customize for your needs**
   - Edit `src/preprocessing.py` for custom preprocessing
   - Edit `src/ocr.py` to add more field patterns
   - Adjust parameters in `ocr_id_card.py`

## 🎓 Learning Resources

- **OpenCV Documentation**: https://docs.opencv.org/
- **Tesseract Documentation**: https://tesseract-ocr.github.io/
- **Tutorial Reference**: `docs/instruction.md` (comprehensive guide)

## 💡 Tips for Best Results

### Image Quality

- Resolution: 300+ DPI for scanned documents
- Lighting: Uniform, no shadows or glare
- Focus: Sharp, clear text
- Angle: Flat, parallel to camera

### Processing Settings

- **Documents/ID Cards**: `--psm 6` (uniform block)
- **Photos**: `--psm 11` (sparse text)
- **Small text**: Increase resize_factor in code
- **Poor lighting**: Use adaptive thresholding

## 📝 Command Reference

```bash
# Show all options
python ocr_id_card.py --help

# Basic processing
python ocr_id_card.py image.jpg

# All options
python ocr_id_card.py image.jpg \
  --lang eng+vie \
  --psm 6 \
  --conf 60 \
  --output results/ \
  --json data.json \
  --quiet
```

## ✅ Implementation Checklist

- [x] Core preprocessing module (grayscale, denoise, binarize, deskew)
- [x] OCR module with confidence filtering
- [x] ID card field extraction with regex patterns
- [x] Bounding box visualization
- [x] CLI application with argparse
- [x] Multiple output formats (PDF, TSV, JSON, images)
- [x] Multi-language support
- [x] Comprehensive documentation
- [x] Installation scripts
- [x] Test utilities
- [x] Example code

## 🎯 Project Success!

Your OCR ID Card recognition application is ready to use! It implements all the best practices from the tutorial including:

- ✅ Professional preprocessing pipeline
- ✅ Robust OCR with Tesseract
- ✅ Confidence filtering and quality control
- ✅ Field extraction and structured output
- ✅ Multiple output formats
- ✅ Comprehensive documentation
- ✅ Easy-to-use CLI interface

**Happy OCR'ing! 🔍📄✨**
