"""
Example script demonstrating how to use the OCR library programmatically.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ocr import ocr_id_card
from src.preprocessing import preprocess_for_ocr
import cv2


def example_basic_usage():
    """Example 1: Basic OCR processing."""
    print("Example 1: Basic OCR Processing")
    print("-" * 60)
    
    # Process ID card image
    results = ocr_id_card(
        image_path="sample_id.jpg",  # Replace with your image
        lang="eng",
        psm=6,
        min_conf=60,
        output_dir="output"
    )
    
    # Print results
    print(f"Recognized Text:\n{results['text']}\n")
    print(f"Extracted Fields:")
    for field, value in results['fields'].items():
        if value:
            print(f"  {field}: {value}")
    
    print(f"\nDeskew Angle: {results['deskew_angle']:.2f}°")
    print(f"Words Detected: {len(results['boxes'])}")


def example_custom_preprocessing():
    """Example 2: Custom preprocessing pipeline."""
    print("\nExample 2: Custom Preprocessing")
    print("-" * 60)
    
    # Load image
    image = cv2.imread("sample_id.jpg")
    
    # Custom preprocessing
    binary, angle = preprocess_for_ocr(
        image,
        resize_factor=2.0,  # More aggressive upsampling
        binarize_method="adaptive",  # Better for photos
        remove_speckles=True
    )
    
    print(f"Preprocessing complete. Rotation corrected: {angle:.2f}°")
    
    # Save preprocessed image
    cv2.imwrite("output/custom_preprocessed.jpg", binary)


def example_multilingual():
    """Example 3: Multi-language OCR."""
    print("\nExample 3: Multi-language OCR")
    print("-" * 60)
    
    # Process Vietnamese ID card with English + Vietnamese
    results = ocr_id_card(
        image_path="vietnam_id.jpg",
        lang="eng+vie",  # Multiple languages
        psm=6,
        min_conf=60,
        output_dir="output"
    )
    
    print(f"Detected text (multilingual):\n{results['text']}")


def example_low_quality_image():
    """Example 4: Processing low-quality images."""
    print("\nExample 4: Low Quality Image Processing")
    print("-" * 60)
    
    # For photos with poor lighting, use adaptive thresholding
    # and lower confidence threshold
    results = ocr_id_card(
        image_path="poor_quality_id.jpg",
        lang="eng",
        psm=11,  # Sparse text mode for challenging images
        min_conf=40,  # Lower threshold to catch more text
        output_dir="output"
    )
    
    # Filter high confidence results
    high_conf_boxes = [box for box in results['boxes'] if box[4] > 70]
    print(f"High confidence words: {len(high_conf_boxes)}/{len(results['boxes'])}")


def example_dataframe_analysis():
    """Example 5: Analyzing OCR data with pandas."""
    print("\nExample 5: DataFrame Analysis")
    print("-" * 60)
    
    results = ocr_id_card(
        image_path="sample_id.jpg",
        lang="eng",
        psm=6,
        min_conf=60,
        output_dir="output"
    )
    
    df = results['dataframe']
    
    # Analyze confidence scores
    print(f"Average confidence: {df['conf'].mean():.2f}")
    print(f"Words with >90% confidence: {len(df[df['conf'] > 90])}")
    
    # Group by line
    lines = df.groupby('line_num')['text'].apply(' '.join)
    print(f"\nText by line:")
    for line_num, text in lines.items():
        print(f"  Line {line_num}: {text}")


if __name__ == "__main__":
    print("OCR ID Card Recognition - Examples")
    print("=" * 60)
    
    # Uncomment the examples you want to run
    # example_basic_usage()
    # example_custom_preprocessing()
    # example_multilingual()
    # example_low_quality_image()
    # example_dataframe_analysis()
    
    print("\nUncomment examples in the script to run them.")
    print("Make sure to replace 'sample_id.jpg' with your actual image path.")
