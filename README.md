# OCR ID Card Recognition

A robust Python application for extracting text from ID cards using OpenCV and Tesseract OCR. This implementation follows best practices from the comprehensive OCR tutorial in `docs/instruction.md`.

## Features

✅ **Complete OCR Pipeline**

- Image preprocessing (grayscale, denoising, binarization)
- Automatic deskewing for rotated images
- High-quality text recognition with confidence filtering
- Bounding box visualization

✅ **ID Card Field Extraction**

- Automatic extraction of common fields (ID number, name, DOB, etc.)
- Pattern matching for various ID formats
- Structured JSON/TSV output

✅ **Multiple Output Formats**

- Searchable PDF with invisible text layer
- Annotated images with bounding boxes
- TSV data for detailed analysis
- JSON structured data

✅ **Robust Preprocessing**

- Adaptive and Otsu thresholding
- Noise removal and speckle cleaning
- Image upsampling for improved accuracy
- Automatic rotation correction

## Installation

### 1. Install Tesseract OCR Engine

**macOS (Homebrew)**

```bash
brew install tesseract
```

**Ubuntu/Debian**

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
# Optional: install additional language packs
sudo apt-get install tesseract-ocr-vie tesseract-ocr-fra
```

**Windows**

```powershell
# Using winget
winget install UB-Mannheim.TesseractOCR

# Or using Chocolatey
choco install tesseract
```

Verify installation:

```bash
tesseract --version
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
# Process an ID card image
python ocr_id_card.py input.jpg

# Specify output directory
python ocr_id_card.py input.jpg --output results/

# Save JSON output
python ocr_id_card.py input.jpg --json output.json
```

### Advanced Usage

```bash
# Multi-language OCR (English + Vietnamese)
python ocr_id_card.py vietnam_id.jpg --lang eng+vie

# Adjust confidence threshold (higher = stricter)
python ocr_id_card.py input.jpg --conf 70

# Different page segmentation modes
python ocr_id_card.py input.jpg --psm 11  # For sparse text
python ocr_id_card.py input.jpg --psm 6   # For uniform blocks (default)

# Quiet mode (no console output)
python ocr_id_card.py input.jpg --quiet
```

## Usage as Python Module

```python
from src.ocr import ocr_id_card

# Process ID card
results = ocr_id_card(
    image_path="id_card.jpg",
    lang="eng",
    psm=6,
    min_conf=60,
    output_dir="output"
)

# Access results
print(results["text"])           # Full recognized text
print(results["fields"])         # Extracted ID fields
print(results["boxes"])          # Bounding boxes
print(results["deskew_angle"])   # Rotation correction angle
```

## Output Files

After processing, the following files are generated in the output directory:

- `<filename>_boxes.jpg` - Image with bounding boxes around detected words
- `<filename>_preprocessed.jpg` - Preprocessed binary image
- `<filename>_ocr.tsv` - Detailed OCR data with confidence scores
- `<filename>.pdf` - Searchable PDF with invisible text layer

## Tesseract Configuration

### Page Segmentation Modes (PSM)

- `3` - Fully automatic (default for general use)
- `4` - Single column of text
- `5` - Single uniform block of vertically aligned text
- `6` - **Single uniform block** (best for ID cards) ⭐
- `7` - Single text line
- `11` - **Sparse text** (good for photos with scattered text)

### OCR Engine Modes (OEM)

- `0` - Legacy engine only
- `1` - LSTM neural network only
- `3` - Default (both engines) ⭐

The application uses `--oem 3` by default for best accuracy.

## Project Structure

```
ocr-quickstart/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Image preprocessing functions
│   └── ocr.py              # OCR and text extraction
├── docs/
│   └── instruction.md      # Comprehensive OCR tutorial
├── ocr_id_card.py          # Main CLI application
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Preprocessing Pipeline

The application follows this preprocessing pipeline for optimal OCR results:

1. **Upsampling** - Resize image by 1.5× for better character recognition
2. **Grayscale** - Convert to grayscale and apply Gaussian blur
3. **Binarization** - Otsu or adaptive thresholding
4. **Noise Removal** - Morphological opening to remove speckles
5. **Deskewing** - Automatic rotation correction

## ID Card Field Extraction

The application automatically extracts common ID card fields:

- ID Number (various formats supported)
- Full Name
- Date of Birth
- Sex/Gender
- Issue Date
- Expiry Date
- Nationality
- Address

Extraction uses regex patterns and context-aware matching.

## Tips for Best Results

### Image Quality

- **Resolution**: Ensure text is at least 20-30 pixels in height
- **Lighting**: Uniform lighting works best
- **Contrast**: High contrast between text and background
- **Focus**: Sharp, in-focus images

### Camera/Scanner Settings

- Use at least 300 DPI for scanned documents
- Avoid shadows and glare
- Keep ID card flat and parallel to camera
- Good lighting from multiple angles

### Troubleshooting

**Low accuracy?**

- Try upsampling: The default 1.5× can be increased
- Use adaptive thresholding for photos with uneven lighting
- Try different PSM modes (6 for documents, 11 for photos)

**Missing text?**

- Lower the confidence threshold: `--conf 40`
- Check if the correct language pack is installed
- Ensure image is not too small (resize before processing)

**Rotated/skewed text?**

- The deskewing algorithm handles ±45° rotation
- For extreme rotation, manually rotate first

## Language Support

Install additional language packs as needed:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr-vie  # Vietnamese
sudo apt-get install tesseract-ocr-fra  # French
sudo apt-get install tesseract-ocr-deu  # German

# macOS (Homebrew)
brew install tesseract-lang

# Then use with --lang flag
python ocr_id_card.py input.jpg --lang eng+vie
```

## Examples

### Example 1: US Driver's License

```bash
python ocr_id_card.py drivers_license.jpg --lang eng --psm 6
```

### Example 2: Vietnamese ID Card

```bash
python ocr_id_card.py vietnam_id.jpg --lang eng+vie --psm 6
```

### Example 3: Photo of ID Card (with glare)

```bash
python ocr_id_card.py photo_id.jpg --lang eng --psm 11 --conf 50
```

## Performance Optimization

- **Speed**: Reduce image size (while maintaining legibility)
- **Accuracy**: Increase resize_factor in preprocessing
- **Memory**: Process images in batches for multiple files

## Dependencies

- **opencv-python** - Image processing
- **pytesseract** - Tesseract Python wrapper
- **Pillow** - Additional image handling
- **pandas** - Data manipulation
- **numpy** - Numerical operations

## License

This project is provided as-is for educational and commercial use.

## Acknowledgments

Based on the comprehensive OCR tutorial: "From Pixels to Paragraphs: A Hands-On OCR Pipeline in Python with OpenCV & Tesseract" (see `docs/instruction.md`).

## Contributing

Contributions are welcome! Areas for improvement:

- Additional ID card format patterns
- More language-specific extraction rules
- Advanced perspective correction
- Deep learning-based text detection (EAST/CRAFT)

## Support

For issues or questions:

1. Check the troubleshooting section
2. Review the comprehensive tutorial in `docs/instruction.md`
3. Verify Tesseract installation and language packs

---

**Happy OCR'ing! 🔍📄✨**
