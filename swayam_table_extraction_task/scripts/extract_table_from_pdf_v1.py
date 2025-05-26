#!/usr/bin/env python3
"""
pdf_to_csv_extractor.py

A robust, fault-tolerant tool to extract all tables from both native and scanned PDF files into CSVs.

Features:
- Auto-detects whether a PDF is native (contains text) or scanned (image-only).
- For native PDFs: uses Camelot (lattice & stream) for high-fidelity table parsing.
- For scanned PDFs: optionally uses a YOLOv8 table-detection model, falls back to full-page OCR.
- Processes all pages of multi-page PDFs.
- Exports each detected table to its own CSV in the output directory.
"""

import os
import argparse
import logging
from pathlib import Path
import os
import logging
import shutil
import tempfile
from retrying import retry
import pdfplumber
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import warnings

# #[WARNING] CropBox missing from /Page, defaulting to MediaBox
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", message="[WARNING] CropBox missing from /Page, defaulting to MediaBox")
# ─────────────────────────────────────────────────────────────────────────────
# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ─────────────────────────────────────────────────────────────────────────────
def is_scanned_pdf(pdf_path: Path) -> bool:
    """
    Heuristic check: if none of the pages contain extractable text,
    assume it's a scanned (image-only) PDF.
    """
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                return False
    return True

@retry(stop_max_attempt_number=3, wait_fixed=1000)
def safe_remove_temp_dir(temp_dir):
    """
    Safely remove temporary directory with retries to handle file-locking issues.
    
    Args:
        temp_dir (str): Path to temporary directory.
    """
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except PermissionError as e:
        logging.warning(f"PermissionError during temp dir cleanup: {e}")
        raise

def extract_native_pdf(pdf_path, output_dir):
    """
    Extract tables from the PDF and save them as CSV files.
    
    Args:
        pdf_path (str): Path to the PDF file.
    """
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp()
    try:
        # Validate PDF
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            if num_pages < 1 or num_pages > 100:
                logging.error(f"Invalid page count ({num_pages}) in {pdf_path}")
                return
        
        logging.info(f"Processing PDF: {pdf_path} ({num_pages} pages)")
        logging.info(f"Detected native PDF: {pdf_path}")
        # Process pages incrementally
        for page_num in range(num_pages):
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    page = pdf.pages[page_num]
                    tables = page.extract_tables()
                    for i, table in enumerate(tables):
                        if table:  # Ensure table is not empty
                            output_file = os.path.join(output_dir, f'{base_name}_table_native_p{page_num+1}_{i}.csv')
                            with open(output_file, 'w', encoding='utf-8') as f:
                                for row in table:
                                    # Handle None values and join row as CSV
                                    safe_row = [str(cell) if cell is not None else '' for cell in row]
                                    f.write(','.join(safe_row) + '\n')
                            logging.info(f"Saved table to {output_file}")
            except Exception as e:
                logging.error(f"Error processing page {page_num+1} of native PDF {pdf_path}: {e}")
    except Exception as e:
        logging.error(f"Error opening PDF {pdf_path}: {e}")
    finally:
        try:
            safe_remove_temp_dir(temp_dir)
        except Exception as e:
            logging.error(f"Failed to clean up temp dir {temp_dir}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
def extract_scanned_tables(
    pdf_path: Path, output_dir: Path, yolo_model_path: str = None
):
    """
    Extract tables from a scanned PDF:
    - If a YOLOv8 table‐detection model is provided, detect & crop each table region.
    - Otherwise, treat the entire page as one region.
    - OCR each region with Tesseract and split lines/columns heuristically.
    """
    logging.info(f"[SCANNED] Processing {pdf_path.name}")
    images = convert_from_path(str(pdf_path), dpi=300)
    
    # Load YOLO model if available
    model = None

    if yolo_model_path and Path(yolo_model_path).exists():
        logging.info(f"[SCANNED] Loading YOLO model from {yolo_model_path}")
        model = YOLO(yolo_model_path)
    else:
        logging.info("[SCANNED] No YOLO model provided or found; using full-page OCR")

    table_counter = 0
    for page_idx, pil_img in enumerate(images, start=1):
        img = np.array(pil_img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        regions = []

        if model:
            # Detect table bounding boxes
            results = model(img)
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                regions.append(gray[y1:y2, x1:x2])
            if not regions:
                regions.append(gray)  # fallback if model finds none
        else:
            regions.append(gray)

        # OCR & save each region as its own CSV
        for region in regions:
            table_counter += 1
            ocr_text = pytesseract.image_to_string(region, config="--psm 6")
            lines = [ln for ln in ocr_text.splitlines() if ln.strip()]

            # Heuristic split: whitespace delim → columns
            rows = [ln.split() for ln in lines]
            if not rows:
                continue

            df = pd.DataFrame(rows)
            out_file = output_dir / f"{pdf_path.stem}_scanned_table_{table_counter}.csv"
            df.to_csv(out_file, index=False, header=False)
            logging.info(f"[SCANNED] Saved table #{table_counter} → {out_file.name}")

# ─────────────────────────────────────────────────────────────────────────────
def process_pdf(pdf_path: Path, output_dir: Path, yolo_model: str = None):
    """
    Main entry for each PDF: detect type then extract accordingly.
    """
    try:
        if is_scanned_pdf(pdf_path):
            extract_scanned_tables(pdf_path, output_dir, yolo_model)
        else:
            extract_native_pdf(pdf_path, output_dir)
    except Exception as e:
        logging.error(f"Error processing {pdf_path.name}: {e}", exc_info=False)

# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract all tables from PDFs into CSV files."
    )
    parser.add_argument(
        "input", help="Path to a PDF file or a directory containing PDFs"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output_tablesV_one",
        help="Directory where CSVs will be saved",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="(Optional) Path to YOLOv8 table-detection .pt weights",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather all PDF files
    pdf_files = [input_path] if input_path.is_file() else list(input_path.glob("*.pdf"))
    if not pdf_files:
        logging.error("No PDF files found at the specified path.")
        return

    for pdf in pdf_files:
        process_pdf(pdf, output_dir, args.model)

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
