# PDF Processing and OCR Enhancement Tool

## Overview

This comprehensive tool processes PDF documents to:
- Detect and correct skew angles
- Rotate misaligned pages
- Perform high-quality OCR using multiple engines (Tesseract, PaddleOCR, EasyOCR)
- Extract text with word-level precision
- Identify special patterns (dates, numbers, codes, etc.)
- Support various table structures
- Generate corrected output PDFs

## Key Features

### 1. Multi-Engine OCR Support
- **Tesseract OCR**: Standard open-source OCR engine
- **PaddleOCR**: High-performance deep learning-based OCR
- **EasyOCR**: Lightweight deep learning OCR
- Automatic engine selection based on performance

### 2. Advanced Skew Detection
- Hough Line Transform for line-based angle detection
- OCR-based structural analysis
- Projection profile analysis
- Contour-based detection
- Weighted consensus for final angle determination

### 3. Comprehensive Text Processing
- Word-level extraction with coordinates
- Confidence scoring for each word
- Special pattern recognition (dates, emails, phone numbers, etc.)
- Table structure detection

### 4. Output Options
- Corrected PDF output
- JSON analysis report
- Text export (TXT, CSV)
- Search functionality within extracted text

## Installation

### Prerequisites
- Python 3.7+
- Tesseract OCR installed (https://github.com/tesseract-ocr/tesseract)
- PDF processing libraries

### Install Required Packages
```bash
pip install -r requirements.txt
```

The tool will automatically detect available OCR engines and use the best options.

## Usage

### Basic Command
```python
python pdf_ocr_processor.py
```

### Configuration
Edit the following variables at the top of the script:
```python
# Input and output file paths
pdf_file = r'path\to\input.pdf'
output_file = r'path\to\output.pdf'

# OCR Engine paths (if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### Advanced Options
After processing, you can access additional functions:
```python
# Search extracted text
search_extracted_text("invoice number")

# Export extracted text
export_extracted_text('txt')  # or 'csv'

# Analyze document structure
analyze_document_structure()
```

## Processing Workflow

1. **PDF Extraction**: Converts each PDF page to high-resolution images
2. **Orientation Correction**: Detects and corrects page rotation
3. **Skew Detection**: Uses multiple methods to determine skew angle
4. **Skew Correction**: Rotates image to correct detected skew
5. **OCR Processing**: Runs multiple OCR engines and selects best results
6. **Pattern Recognition**: Identifies special patterns in extracted text
7. **Output Generation**: Creates corrected PDF and analysis files

## Output Files

- `output.pdf`: Corrected PDF document
- `output_analysis.json`: Detailed OCR results and metrics
- `output_extracted_text.txt`: All extracted text (optional)
- `output_word_details.csv`: Word-level extraction details (optional)

## Performance Notes

- Processing time depends on PDF size and complexity
- Larger documents may require more memory
- For best results, ensure original PDFs are at least 300 DPI
- Table-heavy documents may require additional processing time
