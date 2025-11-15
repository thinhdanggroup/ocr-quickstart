"""
OCR module for ID card text extraction using Tesseract.
Includes confidence filtering and structured data extraction.
"""

import cv2
import pytesseract
from pytesseract import Output
import pandas as pd
import re


def ocr_with_boxes(binary_image, lang="eng", psm=6, min_conf=60):
    """
    Perform OCR and return text with bounding boxes.

    Args:
        binary_image: Preprocessed binary image
        lang: Tesseract language code (e.g., "eng", "eng+vie")
        psm: Page Segmentation Mode (6=uniform block, 11=sparse text)
        min_conf: Minimum confidence threshold (0-100)

    Returns:
        Tuple of (recognized_text, boxes_list, data_dict)
        - recognized_text: Full extracted text
        - boxes_list: [(x, y, w, h, conf, text), ...]
        - data_dict: Raw Tesseract data dictionary
    """
    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(
        binary_image, lang=lang, config=config, output_type=Output.DICT
    )

    boxes = []
    text_chunks = []

    n = len(data["text"])
    for i in range(n):
        conf = int(data["conf"][i])
        txt = data["text"][i]

        if conf > min_conf and txt.strip():
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            boxes.append((x, y, w, h, conf, txt))
            text_chunks.append(txt)

    recognized_text = " ".join(text_chunks)
    return recognized_text, boxes, data


def ocr_to_dataframe(binary_image, lang="eng", psm=6):
    """
    Perform OCR and return results as pandas DataFrame.

    Args:
        binary_image: Preprocessed binary image
        lang: Tesseract language code
        psm: Page Segmentation Mode

    Returns:
        pandas DataFrame with columns: level, page_num, block_num, par_num,
        line_num, word_num, left, top, width, height, conf, text
    """
    config = f"--oem 3 --psm {psm}"
    df = pytesseract.image_to_data(
        binary_image, lang=lang, config=config, output_type=Output.DATAFRAME
    )

    # Clean up DataFrame
    df = df.dropna(subset=["text"]).copy()
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(-1).astype(int)
    df = df[df["text"].str.strip() != ""]

    return df


def extract_id_card_fields(text):
    """
    Extract common ID card fields from OCR text using regex patterns.

    Args:
        text: Full OCR text string

    Returns:
        Dictionary with extracted fields (id_number, name, dob, etc.)
    """
    fields = {
        "id_number": None,
        "name": None,
        "date_of_birth": None,
        "address": None,
        "nationality": None,
        "sex": None,
        "issue_date": None,
        "expiry_date": None,
    }

    # ID Number patterns (various formats)
    # Examples: 123456789, 123-456-789, A1234567, etc.
    id_patterns = [
        r"\b[A-Z]?\d{8,12}\b",  # Alphanumeric ID
        r"\b\d{3}-\d{3}-\d{3}\b",  # Dashed format
        r"ID[:\s]*([A-Z0-9]+)",  # ID: prefix
        r"No[:\s]*([A-Z0-9]+)",  # No: prefix
    ]

    for pattern in id_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            fields["id_number"] = match.group(1) if match.lastindex else match.group(0)
            break

    # Date patterns (DD/MM/YYYY, DD-MM-YYYY, MM/DD/YYYY, etc.)
    date_pattern = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    dates = re.findall(date_pattern, text)

    # Try to identify DOB, issue date, expiry date by context
    if dates:
        for i, date_val in enumerate(dates):
            if i == 0:
                fields["date_of_birth"] = date_val
            elif i == 1:
                fields["issue_date"] = date_val
            elif i == 2:
                fields["expiry_date"] = date_val

    # Sex/Gender (M/F, Male/Female)
    sex_match = re.search(r"\b(Male|Female|M|F)\b", text, re.IGNORECASE)
    if sex_match:
        fields["sex"] = sex_match.group(1).upper()

    # Name extraction (typically after "Name:" label)
    name_match = re.search(
        r"Name[:\s]+([A-Z\s]+?)(?:\n|$|Date|DOB|Sex)", text, re.IGNORECASE
    )
    if name_match:
        fields["name"] = name_match.group(1).strip()

    return fields


def draw_boxes(bgr_image, boxes, color=(0, 255, 0), thickness=2, show_text=True):
    """
    Draw bounding boxes on image with optional text labels.

    Args:
        bgr_image: Input BGR image
        boxes: List of (x, y, w, h, conf, text) tuples
        color: Box color in BGR format
        thickness: Line thickness
        show_text: Whether to overlay recognized text

    Returns:
        Image with drawn boxes
    """
    output = bgr_image.copy()

    for box in boxes:
        x, y, w, h, conf, txt = box

        # Draw rectangle
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)

        # Optionally add text label
        if show_text and txt:
            # Put text above the box
            label = f"{txt} ({conf}%)" if conf < 100 else txt
            cv2.putText(
                output,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    return output


def generate_searchable_pdf(binary_image, output_path, lang="eng", psm=6):
    """
    Generate a searchable PDF with invisible text layer.

    Args:
        binary_image: Preprocessed binary image
        output_path: Path to save PDF file
        lang: Tesseract language code
        psm: Page Segmentation Mode
    """
    config = f"--oem 3 --psm {psm}"
    pdf_bytes = pytesseract.image_to_pdf_or_hocr(
        binary_image, lang=lang, config=config, extension="pdf"
    )

    with open(output_path, "wb") as f:
        f.write(pdf_bytes)


def ocr_id_card(image_path, lang="eng", psm=6, min_conf=60, output_dir="output"):
    """
    Complete OCR pipeline for ID card processing.

    Args:
        image_path: Path to ID card image
        lang: Tesseract language code
        psm: Page Segmentation Mode
        min_conf: Minimum confidence threshold
        output_dir: Directory to save outputs

    Returns:
        Dictionary with:
        - text: Full recognized text
        - fields: Extracted ID card fields
        - boxes: Bounding boxes list
        - dataframe: Full OCR data
        - preprocessed_image: Binary processed image
    """
    import os
    from .preprocessing import preprocess_for_ocr

    # Load image
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Preprocess
    binary, angle = preprocess_for_ocr(
        bgr, resize_factor=1.5, binarize_method="otsu", remove_speckles=True
    )

    # Perform OCR
    text, boxes, data = ocr_with_boxes(binary, lang=lang, psm=psm, min_conf=min_conf)
    df = ocr_to_dataframe(binary, lang=lang, psm=psm)

    # Extract structured fields
    fields = extract_id_card_fields(text)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save TSV
    df.to_csv(f"{output_dir}/{base_name}_ocr.tsv", sep="\t", index=False)

    # Save boxed image
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    boxed_image = draw_boxes(binary_bgr, boxes, show_text=True)
    cv2.imwrite(f"{output_dir}/{base_name}_boxes.jpg", boxed_image)

    # Save preprocessed image
    cv2.imwrite(f"{output_dir}/{base_name}_preprocessed.jpg", binary)

    # Generate searchable PDF
    generate_searchable_pdf(binary, f"{output_dir}/{base_name}.pdf", lang=lang, psm=psm)

    return {
        "text": text,
        "fields": fields,
        "boxes": boxes,
        "dataframe": df,
        "preprocessed_image": binary,
        "deskew_angle": angle,
    }
