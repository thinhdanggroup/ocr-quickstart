#!/usr/bin/env python3
"""
OCR ID Card Recognition CLI
A command-line tool for extracting text from ID cards using Tesseract OCR.
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ocr import ocr_id_card


def print_results(results):
    """Print OCR results in a readable format."""
    print("\n" + "="*60)
    print("OCR RESULTS")
    print("="*60)
    
    print(f"\n📐 Deskew Angle: {results['deskew_angle']:.2f} degrees")
    print(f"📦 Words Detected: {len(results['boxes'])}")
    
    print("\n📄 FULL TEXT:")
    print("-" * 60)
    print(results['text'])
    
    print("\n🆔 EXTRACTED ID CARD FIELDS:")
    print("-" * 60)
    for field, value in results['fields'].items():
        if value:
            print(f"  {field.replace('_', ' ').title()}: {value}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from ID card images using OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python ocr_id_card.py input.jpg
  
  # Specify language (e.g., English + Vietnamese)
  python ocr_id_card.py input.jpg --lang eng+vie
  
  # Adjust confidence threshold
  python ocr_id_card.py input.jpg --conf 70
  
  # Use adaptive thresholding for photos
  python ocr_id_card.py input.jpg --psm 11
  
  # Save JSON output
  python ocr_id_card.py input.jpg --json results.json
        """
    )
    
    parser.add_argument(
        "image",
        help="Path to ID card image file"
    )
    
    parser.add_argument(
        "--lang",
        default="eng",
        help="Tesseract language code (e.g., eng, eng+vie, eng+fra). Default: eng"
    )
    
    parser.add_argument(
        "--psm",
        type=int,
        default=6,
        choices=range(0, 14),
        help="Page Segmentation Mode: 6=uniform block (default, good for ID cards), "
             "11=sparse text, 3=fully automatic"
    )
    
    parser.add_argument(
        "--conf",
        type=int,
        default=60,
        help="Minimum confidence threshold (0-100). Default: 60"
    )
    
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory for results. Default: output/"
    )
    
    parser.add_argument(
        "--json",
        help="Save extracted fields to JSON file"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (only save files)"
    )
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.image):
        print(f"❌ Error: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)
    
    # Run OCR
    try:
        if not args.quiet:
            print(f"🔍 Processing image: {args.image}")
            print(f"   Language: {args.lang}")
            print(f"   PSM: {args.psm}")
            print(f"   Min Confidence: {args.conf}%")
            print(f"   Output: {args.output}/")
            print("\n⏳ Running OCR pipeline...")
        
        results = ocr_id_card(
            image_path=args.image,
            lang=args.lang,
            psm=args.psm,
            min_conf=args.conf,
            output_dir=args.output
        )
        
        # Print results to console
        if not args.quiet:
            print_results(results)
            
            base_name = os.path.splitext(os.path.basename(args.image))[0]
            print("\n✅ Output files saved:")
            print(f"   • {args.output}/{base_name}_boxes.jpg (image with bounding boxes)")
            print(f"   • {args.output}/{base_name}_preprocessed.jpg (preprocessed image)")
            print(f"   • {args.output}/{base_name}_ocr.tsv (detailed OCR data)")
            print(f"   • {args.output}/{base_name}.pdf (searchable PDF)")
        
        # Save JSON if requested
        if args.json:
            json_output = {
                "image": args.image,
                "text": results["text"],
                "fields": results["fields"],
                "deskew_angle": results["deskew_angle"],
                "word_count": len(results["boxes"])
            }
            
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(json_output, f, indent=2, ensure_ascii=False)
            
            if not args.quiet:
                print(f"   • {args.json} (JSON output)")
        
        if not args.quiet:
            print("\n✨ Done!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
