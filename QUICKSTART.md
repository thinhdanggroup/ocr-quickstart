# Quick Start Guide

## Installation

### Option 1: Automatic Installation (macOS/Linux)

```bash
./install.sh
```

This script will:

- Check for Python 3
- Install Tesseract OCR (if not present)
- Create a virtual environment
- Install all Python dependencies

### Option 2: Manual Installation

**Step 1: Install Tesseract**

macOS:

```bash
brew install tesseract
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**Step 2: Install Python Dependencies**

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Command

```bash
python ocr_id_card.py <image_path>
```

### Examples

**Process a US Driver's License:**

```bash
python ocr_id_card.py drivers_license.jpg
```

**Process with Vietnamese language support:**

```bash
# First install Vietnamese language pack
# Ubuntu: sudo apt-get install tesseract-ocr-vie
# macOS: brew install tesseract-lang

python ocr_id_card.py vietnam_id.jpg --lang eng+vie
```

**High confidence filtering:**

```bash
python ocr_id_card.py id_card.jpg --conf 70
```

**Save results as JSON:**

```bash
python ocr_id_card.py id_card.jpg --json results.json
```

## Output

The application creates an `output/` directory with:

1. **`<filename>_boxes.jpg`** - Visual output with bounding boxes
2. **`<filename>_preprocessed.jpg`** - Preprocessed binary image
3. **`<filename>_ocr.tsv`** - Detailed OCR data (Excel-compatible)
4. **`<filename>.pdf`** - Searchable PDF

## Common Issues

### "tesseract: command not found"

Install Tesseract OCR (see Installation section above)

### Low accuracy results

- Use higher resolution images (300+ DPI)
- Ensure good lighting and focus
- Try: `--conf 50` to lower confidence threshold
- Try: `--psm 11` for photos with scattered text

### Wrong language detected

- Install the correct language pack
- Specify language: `--lang eng+vie`

### Image too small

- The application automatically upsamples by 1.5×
- For very small text, edit `src/ocr.py` and increase `resize_factor`

## Next Steps

- Read the full tutorial: `docs/instruction.md`
- Check example scripts: `examples.py`
- Review comprehensive documentation: `README.md`

## Help

```bash
python ocr_id_card.py --help
```

For detailed options and examples.
