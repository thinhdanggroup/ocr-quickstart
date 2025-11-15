# From Pixels to Paragraphs: A Hands‑On OCR Pipeline in Python with OpenCV & Tesseract

Optical Character Recognition (OCR) is one of those quietly magical pieces of software engineering. You point it at a photo or a scan, and—poof—out come words you can search, copy, analyze, and feed to other systems. In this tutorial, we’ll build an end‑to‑end OCR pipeline in Python using **OpenCV** (for image processing) and **Tesseract** (for the actual recognition). We’ll start at installation and finish by **drawing bounding boxes around detected words**, with clear code you can adapt to documents, receipts, screenshots, or “text in the wild.”

If you’re the type who learns by doing, you’re in the right place. By the end, you’ll have a practical toolkit: pre‑processing steps (grayscale, denoising, binarization), deskewing, segmentation, confidence filtering, and Tesseract configuration (PSM/OEM) explained. And yes—lots of commented code.

---

## TL;DR Quickstart

Want a working snippet first? Here’s a minimal example that reads an image, does a bit of pre‑processing, and OCRs English text with Tesseract:

```python
import cv2
import pytesseract

# If you're on Windows, set the Tesseract executable path, e.g.:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = cv2.imread("input.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Basic denoise + Otsu binarization
gray = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

config = r"--oem 3 --psm 6"   # LSTM engine; assume a block of text
text = pytesseract.image_to_string(thresh, lang="eng", config=config)

print(text)
```

That’s the hello‑world. The rest of this post is about turning this into a **robust** pipeline that works on messy real‑world images.

---

## 1) Installing Tesseract, OpenCV, and PyTesseract

You need **three** pieces:

* **Tesseract**: the OCR engine (native binary)
* **pytesseract**: the Python wrapper that calls Tesseract
* **opencv-python**: for image loading and pre/post-processing

**Install the Python packages**:

```bash
pip install opencv-python pytesseract Pillow
# For headless environments (servers/containers), use:
# pip install opencv-python-headless pytesseract Pillow
```

**Install the Tesseract binary** (choose your OS):

* **macOS (Homebrew)**

  ```bash
  brew install tesseract
  ```
* **Ubuntu/Debian**

  ```bash
  sudo apt-get update
  sudo apt-get install tesseract-ocr
  # Optional: install additional language packs (e.g., French and Vietnamese)
  sudo apt-get install tesseract-ocr-fra tesseract-ocr-vie
  ```
* **Windows**

  * Install via the official installer (Tesseract OCR for Windows) or via package manager, e.g.:

    ```powershell
    winget install UB-Mannheim.TesseractOCR
    # or: choco install tesseract
    ```
  * Then tell pytesseract where to find `tesseract.exe`:

    ```python
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    ```

**Verification**:

```bash
tesseract --version
```

If that prints a version, you’re good to go.

---

## 2) The OCR Pipeline, Conceptually

Before diving into code, let’s align on the mental model. Tesseract performs best when you feed it **clean, high‑contrast text regions**. Your job (with OpenCV) is to prepare the image so that Tesseract has an easy day:

1. **Normalize** (resize, grayscale, denoise).
2. **Binarize** (turn pixels clearly into “ink” vs “paper”).
3. **Deskew** (rotate slightly tilted pages).
4. **Segment** (optionally isolate lines/words/regions).
5. **Recognize** (Tesseract with the right settings).
6. **Post‑process** (confidence thresholds, bounding boxes, TSV/HOCR/PDF output).

Think of it like barista + espresso machine: OpenCV grinds and tamps; Tesseract pulls the shot.

---

## 3) Pre‑Processing 101: Grayscale, Denoise, Binarize

### 3.1 Grayscale & Denoising

Color rarely helps OCR. Convert to grayscale, then denoise. A small Gaussian blur is fast and typically sufficient:

```python
import cv2

def to_grayscale_denoise(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Light blur removes high-frequency noise without overly softening characters
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray
```

### 3.2 Binarization: Otsu vs. Adaptive

* **Otsu’s threshold**: great for scans with uniform lighting.
* **Adaptive threshold**: better when illumination varies (photos, shadows).

```python
def binarize(gray, method="otsu"):
    if method == "otsu":
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return thresh
    elif method == "adaptive":
        # Adaptive Gaussian thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=31, C=10
        )
        return thresh
    else:
        raise ValueError("method must be 'otsu' or 'adaptive'")
```

**Tip:** Tesseract generally prefers **black text on white background**. If your image ends up inverted, flip it with `cv2.bitwise_not()`.

---

## 4) Deskewing: Straighten First, Read Later

Slight rotations (±2–5°) can tank accuracy. A classic approach is to compute the angle of the “ink” pixels and rotate the image back.

```python
import numpy as np
import cv2

def deskew(binary):
    # Expect white background, black text. If not, invert:
    # binary = cv2.bitwise_not(binary)

    coords = np.column_stack(np.where(binary == 0))  # positions of ink (0 if text is black)
    if len(coords) == 0:
        # fallback: treat as no skew if no ink detected
        return binary, 0.0

    angle = cv2.minAreaRect(coords)[-1]
    # cv2 returns angle in [-90, 0); map to the smallest absolute rotation
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = binary.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle
```

**When to deskew?** For document‑like images (scans, PDFs, receipts) it almost always helps. For natural scene text (signs, billboards), perspective correction is often more important; see the “Recipes” section.

---

## 5) Segmentation with OpenCV: Find Regions, Lines, and Words

You can let Tesseract figure out layout (it’s good at it), or you can give it a head start by segmenting text blocks. Morphological operations help group nearby characters into lines or paragraphs.

### 5.1 Grouping into lines (morphological dilation)

```python
def find_line_boxes(binary):
    # Binary should be black text on white background; invert for morphology convenience.
    inv = cv2.bitwise_not(binary)
    # Horizontal kernel: connect characters into lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    dilated = cv2.dilate(inv, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]  # (x, y, w, h)
    # Sort top-to-bottom
    boxes = sorted(boxes, key=lambda b: b[1])
    return boxes
```

### 5.2 Grouping into words (smaller kernel)

```python
def find_word_boxes(binary):
    inv = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dilated = cv2.dilate(inv, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))  # top-to-bottom, then left-to-right
    return boxes
```

**Pro‑move:** Instead of OCRing the entire page, pass **cropped ROIs** (per line/word box) to Tesseract. This reduces confusion from non‑text regions and often boosts accuracy.

---

## 6) Let Tesseract Segment for You: `image_to_data` and Confidence

Tesseract can return **word-level results** with bounding boxes and confidence scores via `image_to_data`. This is the simplest way to draw boxes accurately and filter low-confidence hits.

```python
from pytesseract import Output

def ocr_with_boxes(img_bgr, lang="eng", psm=6, min_conf=60):
    gray = to_grayscale_denoise(img_bgr)
    binary = binarize(gray, method="otsu")
    binary, angle = deskew(binary)

    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(binary, lang=lang, config=config, output_type=Output.DICT)
    # data has keys: 'level','page_num','block_num','par_num','line_num','word_num',
    # 'left','top','width','height','conf','text'

    boxes = []
    text_chunks = []
    n = len(data["text"])
    for i in range(n):
        conf = int(data["conf"][i])
        txt = data["text"][i]
        if conf > min_conf and txt.strip():
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            boxes.append((x, y, w, h, conf, txt))
            text_chunks.append(txt)

    recognized_text = " ".join(text_chunks)
    return recognized_text, boxes, binary, angle
```

**Drawing the boxes:**

```python
def draw_boxes(img_bgr, boxes, color=(0, 255, 0), thickness=2):
    out = img_bgr.copy()
    for (x, y, w, h, conf, txt) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        # Optional: label with text or confidence
        cv2.putText(out, txt, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out
```

---

## 7) Tesseract Configuration: OEM and PSM without the Acronyms

Two Tesseract settings matter a lot:

* **OEM** (OCR Engine Mode):

  * `0` = Legacy engine only
  * `1` = LSTM engine only (neural network; modern, generally best)
  * `2` = Legacy + LSTM
  * `3` = Default (whatever is available)
* **PSM** (Page Segmentation Mode):

  * `3` = Fully automatic page segmentation (typical default)
  * `4/5/6` = Treat as a single column/block of text (great for documents and receipts)
  * `7` = Single text line
  * `8` = Single word
  * `11` = Sparse text (no particular order)
  * `12` = Sparse text with OSD (orientation & script detection)
  * `13` = Raw line (no layout analysis)

**Rules of thumb**:

* **Documents** (scans, receipts): `--oem 3 --psm 6`
* **Screenshots** (uniform blocks): `--oem 3 --psm 6` or `--psm 4/5` depending on layout
* **Scene text** (signs, photos): `--oem 3 --psm 11` (sparse text) often works better
* **Single line widgets** (captchas, serials): `--oem 1 --psm 7` with a whitelist

Whitelisting can constrain recognition to known character sets:

```python
config = r'--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
```

Use whitelists judiciously; they help when you *know* your domain (e.g., license plates, IDs).

---

## 8) End‑to‑End Example: Installation to Bounding Boxes

Let’s put it together in a script that:

1. Loads an image
2. Preprocesses (grayscale → binarize → deskew)
3. OCRs with Tesseract
4. Draws word boxes for confident hits
5. Saves the overlay image and a TSV of results

```python
import cv2
import pytesseract
from pytesseract import Output
import pandas as pd

# Uncomment and set on Windows:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_for_ocr(bgr, binarize_method="otsu", resize_factor=1.5):
    # 1) Normalize size (Tesseract is sensitive to text scale)
    h, w = bgr.shape[:2]
    bgr_up = cv2.resize(bgr, (int(w*resize_factor), int(h*resize_factor)), interpolation=cv2.INTER_CUBIC)

    # 2) Grayscale + denoise
    gray = cv2.cvtColor(bgr_up, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3) Binarize
    if binarize_method == "adaptive":
        bin_img = cv2.adaptiveThreshold(gray, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, 10)
    else:
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4) Deskew
    bin_img, angle = deskew(bin_img)
    return bin_img, angle

def run_ocr(img_path, lang="eng", psm=6, min_conf=60, out_prefix="out"):
    bgr = cv2.imread(img_path)
    assert bgr is not None, f"Could not read image: {img_path}"

    binary, angle = preprocess_for_ocr(bgr, binarize_method="otsu", resize_factor=1.5)
    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(binary, lang=lang, config=config, output_type=Output.DICT)

    rows = []
    boxes = []
    for i in range(len(data["text"])):
        txt = data["text"][i]
        conf = int(data["conf"][i])
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        level = data["level"][i]
        rows.append({
            "level": level, "text": txt, "conf": conf, "left": x, "top": y, "width": w, "height": h
        })
        if conf > min_conf and txt.strip():
            boxes.append((x, y, w, h, conf, txt))

    # Draw boxes on the preprocessed (deskewed) image for visual alignment
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    overlay = draw_boxes(binary_bgr, boxes, color=(0, 255, 0), thickness=2)

    # Save results
    cv2.imwrite(f"{out_prefix}_boxes.jpg", overlay)

    df = pd.DataFrame(rows)
    df.to_csv(f"{out_prefix}_ocr.tsv", sep="\t", index=False)

    print(f"Deskew angle: {angle:.2f} degrees")
    print(f"Boxes drawn: {len(boxes)}")
    return df

# Example usage:
# df = run_ocr("input.jpg", lang="eng", psm=6, min_conf=60, out_prefix="result")
```

Run this on a clean scan or screenshot; you’ll get an image with word boxes and a TSV you can analyze.

---

## 9) Advanced Output: Searchable PDFs and HOCR

Sometimes you want a **searchable PDF** with an invisible text layer aligned to your image. PyTesseract can make this directly:

```python
pdf_bytes = pytesseract.image_to_pdf_or_hocr(binary, lang="eng", config="--oem 3 --psm 6", extension="pdf")
with open("output.pdf", "wb") as f:
    f.write(pdf_bytes)
```

For HTML with detailed layout info (bounding boxes, baselines), request **HOCR**:

```python
hocr = pytesseract.image_to_pdf_or_hocr(binary, lang="eng", config="--oem 3 --psm 6", extension="hocr")
with open("layout.hocr", "wb") as f:
    f.write(hocr)
```

HOCR is great for debugging or building custom viewers.

---

## 10) Accuracy & Performance: Practical Tuning

A few levers make outsized differences:

* **Scale matters**: Upsample images by **1.5–2×** (`INTER_CUBIC`). Tesseract’s LSTM engine is sensitive to character height; too small and it guesses.
* **Choose the right PSM**: Documents love `psm 6`; scene text often prefers `psm 11`.
* **Language packs**: OCRing French invoices? Install `tesseract-ocr-fra` or the equivalent and pass `lang="fra"`. Multi‑lang works: `lang="eng+fra"`.
* **Whitelist/blacklist**: If you’re extracting **serial numbers** or **invoice totals**, constrain the character set.
* **ROI first**: Don’t feed whole pictures; crop to relevant regions (e.g., total amount field).
* **Denoise, but don’t over‑blur**: Over‑smoothing erases serifs and thin strokes.
* **Keep colors simple**: High contrast is king. Invert if needed so text is black on white.

For **speed**, reduce image size (to the minimum that preserves legibility), and avoid re‑running full page analysis when you can segment once and OCR many small ROIs.

---

## 11) Recipes for Real‑World Images

### 11.1 Receipts & Invoices (dense text)

* Use **Otsu** binarization and **deskew**.
* `--psm 6` (single uniform block) is a strong default.
* Remove speckles with morphological **opening**.

```python
def clean_receipt(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened
```

### 11.2 Screenshots (UI text)

* Often perfectly uniform; **Otsu** works.
* `--psm 6` or `--psm 4/5` (single column/block).
* Consider **whitelisting** if you only need digits or ASCII.

### 11.3 Scene Text (signs, photos)

* Use **adaptive** thresholding; illumination varies.
* Consider a **morphological gradient** to highlight strokes:

```python
def gradient_then_thresh(gray):
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    _, thr = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thr
```

* `--psm 11` (sparse text) often wins.
* Perspective distortion? Find the four corners of a rectangular sign and **warp** with `cv2.getPerspectiveTransform`.

### 11.4 Rotated or Upside‑Down Text

Tesseract can estimate orientation:

```python
osd = pytesseract.image_to_osd(binary)
print(osd)  # Contains rotation and script info
```

You can parse the rotation angle and rotate the image before the full OCR pass.

---

## 12) Bounding Boxes: Contours vs. Tesseract Data

You have two solid strategies:

1. **Contours first, OCR later**
   Use OpenCV to find likely text regions (morphology + contours), crop ROIs, then OCR each ROI. You get control over *what* you OCR.
2. **OCR first, then boxes from Tesseract**
   Call `image_to_data` and draw boxes returned by Tesseract, filtering by confidence. This tends to be cleaner for documents and screenshots because Tesseract’s notion of “where the text is” matches its recognition.

Here’s a hybrid helper that tries both: use Tesseract boxes if they’re plentiful; otherwise fall back to OpenCV word boxes.

```python
def robust_boxes(img_bgr, prefer_tesseract=True):
    gray = to_grayscale_denoise(img_bgr)
    binary = binarize(gray, method="otsu")
    binary, _ = deskew(binary)

    tess_boxes = []
    if prefer_tesseract:
        data = pytesseract.image_to_data(binary, output_type=Output.DICT)
        for i in range(len(data["text"])):
            conf = int(data["conf"][i])
            txt = data["text"][i]
            if conf > 60 and txt.strip():
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                tess_boxes.append((x, y, w, h, conf, txt))

    if tess_boxes:
        return tess_boxes, binary

    # Fallback: OpenCV-based word boxes
    cv_boxes_raw = find_word_boxes(binary)
    cv_boxes = [(x, y, w, h, 100, "") for (x, y, w, h) in cv_boxes_raw]
    return cv_boxes, binary
```

---

## 13) Structured Outputs: TSV to DataFrame

Tesseract’s TSV is easy to analyze with pandas:

```python
from pytesseract import Output
import pandas as pd

def ocr_to_dataframe(image, lang="eng", psm=6):
    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=Output.DATAFRAME)
    # Columns include: level, page_num, block_num, par_num, line_num, word_num,
    # left, top, width, height, conf, text
    df = data.dropna(subset=["text"]).copy()
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(-1).astype(int)
    return df

# Example:
# df = ocr_to_dataframe(binary)
# high_conf = df[df.conf > 70]
```

With the DataFrame, you can compute **per‑line aggregates**, extract **numeric totals**, or feed the text into downstream NLP pipelines.

---

## 14) Common Pitfalls (and How to Avoid Them)

* **Tiny text**: Upscale. Aim for character heights of at least ~20–30 pixels.
* **Noisy backgrounds**: Try adaptive thresholding; experiment with morphological **opening** to remove small blobs.
* **Colored text**: Convert to grayscale, but consider a **color channel pick** if one channel has higher contrast (e.g., use the `S` channel in HSV for neon signs).
* **Curvy lines**: Deskew helps for global tilt, but wavy baselines (curved pages) are tougher; crop into smaller ROIs where the local line is straight.
* **Wrong PSM**: If Tesseract misses words or smears sentences together, your PSM might be off. Switch between `6` (block) and `11` (sparse) to test.
* **Language mismatch**: If you see gibberish for accented characters, install the correct language pack and pass `lang="eng+fra"` etc.

---

## 15) A Bit of Context: Why Tesseract Works (and When It Doesn’t)

Tesseract started at HP in the late ’80s/early ’90s and was later open‑sourced. Modern Tesseract (v4+) uses an **LSTM‑based neural network** pipeline that’s far more accurate on natural fonts than the legacy engine, but it still expects **readable, high‑contrast text**. It’s not a panacea for low‑resolution, heavily compressed, or wildly perspective‑distorted images. For those, pair OpenCV pre‑processing with smarter region detection—or consider specialized deep‑learning text detectors (EAST/CRAFT) feeding ROIs into Tesseract for recognition.

---

## 16) Putting It All Together: A Reusable CLI

Here’s a small command‑line tool you can drop in a project to OCR an image, save a searchable PDF, and output a boxed preview.

```python
#!/usr/bin/env python3
# ocr_cli.py
import argparse, cv2, pytesseract
from pytesseract import Output
import pandas as pd

def preprocess(image_path, resize=1.75, adaptive=False):
    bgr = cv2.imread(image_path)
    assert bgr is not None, f"Cannot read {image_path}"
    h, w = bgr.shape[:2]
    bgr = cv2.resize(bgr, (int(w*resize), int(h*resize)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    if adaptive:
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, 10)
    else:
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img, angle = deskew(bin_img)
    return bgr, bin_img, angle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to input image")
    ap.add_argument("--lang", default="eng", help="tesseract language(s), e.g. eng+fra")
    ap.add_argument("--psm", type=int, default=6, help="page segmentation mode")
    ap.add_argument("--conf", type=int, default=60, help="min confidence for boxes")
    ap.add_argument("--adaptive", action="store_true", help="use adaptive thresholding")
    ap.add_argument("--out", default="ocr_out", help="output prefix")
    args = ap.parse_args()

    bgr, binary, angle = preprocess(args.image, adaptive=args.adaptive)
    print(f"Deskew angle: {angle:.2f} degrees")

    config = f"--oem 3 --psm {args.psm}"
    df = pytesseract.image_to_data(binary, lang=args.lang, config=config, output_type=Output.DATAFRAME)
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(-1).astype(int)
    df.to_csv(f"{args.out}.tsv", sep="\t", index=False)

    # Draw high-confidence boxes
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    sub = df[(df.conf >= args.conf) & (df.text.notnull()) & (df.text.str.strip()!="")]
    for _, row in sub.iterrows():
        x, y, w, h = int(row.left), int(row.top), int(row.width), int(row.height)
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0,255,0), 2)
    cv2.imwrite(f"{args.out}_boxes.jpg", vis)

    # Also export a searchable PDF
    pdf = pytesseract.image_to_pdf_or_hocr(binary, lang=args.lang, config=config, extension="pdf")
    with open(f"{args.out}.pdf", "wb") as f:
        f.write(pdf)

if __name__ == "__main__":
    main()
```

Usage:

```bash
python ocr_cli.py input.jpg --lang eng --psm 6 --conf 70 --out mydoc
```

You’ll get `mydoc.tsv`, `mydoc_boxes.jpg`, and `mydoc.pdf`.

---

## 17) Section Recap

* **Pre‑processing** is half the battle: grayscale, denoise, binarize (Otsu for documents, adaptive for photos), and **deskew**.
* **Segmentation** helps: use morphological dilation to group lines/words, or rely on Tesseract’s own `image_to_data`.
* **Tesseract settings** matter: choose `PSM` to match layout and `OEM` to use the modern LSTM engine.
* **Bounding boxes** are easy with `image_to_data` (and confidence filtering). Draw rectangles via OpenCV to visualize.
* **Structured outputs** (TSV/HOCR/PDF) let you analyze, debug, and integrate OCR results downstream.

---

## 18) Further Reading & Next Steps

* **OpenCV docs**: Morphological ops, thresholding, and geometric transforms are your bread and butter for OCR pre‑processing.
* **Tesseract user guide**: Deep dives on language models, training, PSM/OEM nuances, and configuration options.
* **Advanced pipelines**: Combine text detectors (EAST/CRAFT/DB) for robust region detection, then hand ROIs to Tesseract for recognition.
* **Post‑processing**: Regular expressions to extract totals, dates, IDs; simple spell‑correction; or domain dictionaries to fix OCR “near misses.”

---

### Key Takeaway

Practical OCR in Python is about **pairing Tesseract with the right image processing**: grayscale → denoise → binarize → (optionally) deskew & segment → OCR with appropriate `psm`/`oem`. With a few well‑chosen steps, you get reliable text extraction **and** trustworthy bounding boxes—ready for search, analytics, or automation.

Happy OCR’ing!
