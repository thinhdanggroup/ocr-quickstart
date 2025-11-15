"""
Image preprocessing module for OCR.
Includes grayscale conversion, denoising, binarization, and deskewing.
"""

import cv2
import numpy as np


def to_grayscale_denoise(bgr_image):
    """
    Convert image to grayscale and apply Gaussian blur to denoise.
    
    Args:
        bgr_image: BGR color image (OpenCV format)
        
    Returns:
        Grayscale denoised image
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    # Light blur removes high-frequency noise without overly softening characters
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def binarize(gray_image, method="otsu"):
    """
    Convert grayscale image to binary (black text on white background).
    
    Args:
        gray_image: Grayscale image
        method: "otsu" for uniform lighting (ID cards, scans) or 
                "adaptive" for variable lighting
                
    Returns:
        Binary image
    """
    if method == "otsu":
        _, thresh = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return thresh
    elif method == "adaptive":
        # Adaptive Gaussian thresholding for variable lighting
        thresh = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=31, C=10
        )
        return thresh
    else:
        raise ValueError("method must be 'otsu' or 'adaptive'")


def deskew(binary_image):
    """
    Detect and correct image skew/rotation.
    
    Args:
        binary_image: Binary image (black text on white background)
        
    Returns:
        Tuple of (deskewed_image, angle_in_degrees)
    """
    # Find coordinates of all black pixels (text)
    coords = np.column_stack(np.where(binary_image == 0))
    
    if len(coords) == 0:
        # No text detected, return as-is
        return binary_image, 0.0
    
    # Calculate minimum area rectangle around text
    angle = cv2.minAreaRect(coords)[-1]
    
    # Normalize angle to smallest absolute rotation
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Rotate image
    (h, w) = binary_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary_image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated, angle


def remove_noise(binary_image, kernel_size=(1, 1)):
    """
    Remove small speckles using morphological opening.
    Useful for cleaning receipts and ID cards.
    
    Args:
        binary_image: Binary image
        kernel_size: Size of morphological kernel
        
    Returns:
        Cleaned binary image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened


def preprocess_for_ocr(bgr_image, resize_factor=1.5, binarize_method="otsu", 
                       remove_speckles=True):
    """
    Complete preprocessing pipeline for OCR.
    
    Args:
        bgr_image: Input BGR image
        resize_factor: Scale factor for upsampling (improves OCR for small text)
        binarize_method: "otsu" or "adaptive"
        remove_speckles: Whether to apply noise removal
        
    Returns:
        Tuple of (preprocessed_image, deskew_angle)
    """
    # 1) Normalize size - Tesseract is sensitive to text scale
    h, w = bgr_image.shape[:2]
    bgr_upsampled = cv2.resize(
        bgr_image, 
        (int(w * resize_factor), int(h * resize_factor)), 
        interpolation=cv2.INTER_CUBIC
    )
    
    # 2) Grayscale + denoise
    gray = to_grayscale_denoise(bgr_upsampled)
    
    # 3) Binarize
    binary = binarize(gray, method=binarize_method)
    
    # 4) Remove speckles (optional, good for ID cards)
    if remove_speckles:
        binary = remove_noise(binary)
    
    # 5) Deskew
    binary, angle = deskew(binary)
    
    return binary, angle


def find_text_regions(binary_image, kernel_width=40, kernel_height=1):
    """
    Find text line regions using morphological dilation.
    
    Args:
        binary_image: Binary image (black text on white background)
        kernel_width: Width of horizontal kernel (larger = connect more characters)
        kernel_height: Height of kernel
        
    Returns:
        List of bounding boxes [(x, y, w, h), ...] sorted top-to-bottom
    """
    # Invert for morphology (white text on black background)
    inverted = cv2.bitwise_not(binary_image)
    
    # Horizontal kernel to connect characters into lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
    dilated = cv2.dilate(inverted, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes and sort top-to-bottom
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = sorted(boxes, key=lambda b: b[1])
    
    return boxes
