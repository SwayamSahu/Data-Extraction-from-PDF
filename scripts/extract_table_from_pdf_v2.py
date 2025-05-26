#!/usr/bin/env python3
"""
pdf_to_csv_extractor.py

A robust, fault-tolerant tool to extract all tables from both native and scanned PDF files into CSVs.

Features:
- Auto-detects whether a PDF is native (contains text) or scanned (image-only).
- For native PDFs: uses pdfplumber for high-fidelity table parsing.
- For scanned PDFs: uses Microsoft's Table Transformer detection model and Tesseract OCR.
- Processes all pages of multi-page PDFs.
- Exports each detected table to its own CSV in the output directory.
"""

import os
import fitz
import torch
import argparse
import logging
from pathlib import Path
import shutil
import tempfile
from retrying import retry
import pdfplumber
from img2table.document import PDF
import pdfplumber
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
from PIL import Image
import pytesseract
import cv2
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

OCR_CONFIG = r"--oem 3 --psm 6" # Tesseract config for OCR
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
detection_model = (
    TableTransformerForObjectDetection
    .from_pretrained("microsoft/table-transformer-detection")
    .to(device)
)

# ─────────────────────────────────────────────────────────────────────────────

# def is_scanned_pdf(pdf_path):
#     with pdfplumber.open(pdf_path) as pdf:
#         for i, page in enumerate(pdf.pages):
#             tables = page.extract_tables()
#             if tables:
#                 print(f"\nPage {i+1} has tables.")
#                 for table_index, table in enumerate(tables):
#                     has_text = any(any(cell and cell.strip() for cell in row) for row in table)
#                     if has_text:
#                         print(f"  ✅ Table {table_index+1} has native text.")
#                         return False
#                     else:
#                         print(f"  ❌ Table {table_index+1} likely scanned or has no extractable text.")
#                         return True
#             else:
#                 print(f"\nPage {i+1} has no tables.")
#                 return True


def is_scanned_pdf(pdf_path: Path) -> bool:
    """
    Determine if a PDF is scanned by checking for extractable text on any page.
    Heuristic check: if none of the pages contain extractable text,
    assume it's a scanned (image-only) PDF.

    Args:
        pdf_path: Path to the input PDF file.
    Returns:
        True if PDF appears scanned (image-based), False otherwise.
    """
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                return False
    return True

def pdf_to_images(pdf_path, dpi=300):
    """
    Render each PDF page to a PIL image via PyMuPDF.
    Convert each page of the PDF into high-resolution images using PyMuPDF.

    Args:
        pdf_path: Path to PDF file.
        dpi: Resolution for rendering.
    Returns:
        List of PIL images, one per page.
    """
    imgs = []
    doc = fitz.open(pdf_path)
    zoom = dpi / 72  # 72 dpi is the default resolution in PDF
    mat = fitz.Matrix(zoom, zoom)
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        imgs.append(img)
    doc.close()
    return imgs

def detect_tables(image: Image.Image):
    """
    Return list of cropped PIL images of each detected table.

    Use Microsoft's Table Transformer model to detect tables in an image.
    
    Args:
        image: PIL image of a PDF page.
    Returns:
        List of cropped PIL images of each detected table.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = detection_model(**inputs)
    # post-process: threshold + resize to original
    results = processor.post_process_object_detection(
        outputs, threshold=0.8, target_sizes=[image.size[::-1]]
    )[0]

    tables = []
    for box, label in zip(results["boxes"], results["labels"]):
        if detection_model.config.id2label[label.item()] == "table":
            x0, y0, x1, y1 = [int(b) for b in box]
            tables.append(image.crop((x0, y0, x1, y1)))
    return tables

def ocr_table(image: Image.Image):
    """
    OCR the cropped table image into a DataFrame of whitespace-split cells.
    OCR a cropped table image into a pandas DataFrame.
    
    Args:
        image: Cropped table as a PIL image.
    Returns:
        DataFrame of extracted text structured in rows and columns.
    """
    # convert to OpenCV BGR
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    text = pytesseract.image_to_string(cv_img, config=OCR_CONFIG)
    # split into lines, then whitespace-split each line
    lines = [ln for ln in text.splitlines() if ln.strip()]
    rows = [ln.split() for ln in lines]
    if not rows:
        return pd.DataFrame()
    # normalize ragged rows
    max_cols = max(len(r) for r in rows)
    rows_padded = [r + [""] * (max_cols - len(r)) for r in rows]
    return pd.DataFrame(rows_padded)

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
    Extract tables from a native (text-based) PDF using pdfplumber and save them as CSVs.
    
    Args:
        pdf_path: Path to input PDF.
        output_dir: Directory to store the output CSVs.
    """
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp()
    try:
        # Validate PDF
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            if num_pages < 1 or num_pages > 2000:
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

def extract_scanned_pdf(pdf_path: Path, output_folder: Path):
    """
    Detect tables in scanned (image-based) PDFs using Table Transformer + OCR.
    
    Args:
        pdf_path: Path to scanned PDF file.
        output_folder: Directory to store the output CSVs.
    """
    try:
        logging.info(f"[SCANNED] Processing {pdf_path.name}")
        file_name = os.path.basename(pdf_path)
        images = pdf_to_images(pdf_path)
        extracted = []
        for page_idx, img in enumerate(images, start=1):
            print(f"[Page {page_idx}] rendering → detect tables…")
            tables = detect_tables(img)

            if not tables:
                print(f"  no tables found.")
                continue

            for tbl_idx, tbl_img in enumerate(tables, start=1):
                print(f"  [Table {tbl_idx}] OCR’ing…")
                df = ocr_table(tbl_img)
                extracted.append((page_idx, tbl_idx, df))

                # save CSV
                csv_name = f"{file_name}_page_{page_idx}_table_{tbl_idx}.csv"
                path = os.path.join(output_folder, csv_name)
                df.to_csv(path, index=False, header=False)
                print(f"    → saved {csv_name} ({df.shape[0]}×{df.shape[1]})")
        print(f"\nDone. Extracted {len(tables)} tables total.")
    except Exception as e:
        logging.error(f"Error processing {pdf_path.name}: {e}", exc_info=False)

# ─────────────────────────────────────────────────────────────────────────────
def process_pdf(pdf_path: Path, output_dir: Path):
    """
    Main processing function to determine PDF type and run appropriate extractor.
    
    Args:
        pdf_path: PDF to process.
        output_dir: Directory to save output CSVs.
    """
    try:
        if is_scanned_pdf(pdf_path):
            extract_scanned_pdf(pdf_path, output_dir)
        else:
            extract_native_pdf(pdf_path, output_dir)
    except Exception as e:
        logging.error(f"Error processing {pdf_path.name}: {e}", exc_info=False)

# ─────────────────────────────────────────────────────────────────────────────
def main():
    """
    Command-line interface for extracting tables from PDFs.
    Supports individual files or directories.
    """
    parser = argparse.ArgumentParser(
        description="Extract all tables from PDFs into CSV files."
    )
    parser.add_argument(
        "input", help="Path to a PDF file or a directory containing PDFs"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output_tables4",
        help="Directory where CSVs will be saved",
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
        process_pdf(pdf, output_dir)

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
