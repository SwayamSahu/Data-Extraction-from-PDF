import os
import logging
import time
import shutil
import tempfile
from retrying import retry
import pdfplumber
from img2table.document import PDF
from img2table.ocr import TesseractOCR

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/extraction.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set Tesseract executable path (update this with your actual Tesseract path)
TESSERACT_PATH = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  

# Set tessdata directory (directory containing 'tessdata' folder)
TESSDATA_DIR = r'C:/Program Files/Tesseract-OCR/tessdata'  

def is_native_pdf(pdf_path):
    """
    Determine if the PDF is native (contains font resources) or scanned.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        bool: True if native, False if scanned.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                if page.chars:  # Check if page contains extractable text
                    return True
            return False
    except Exception as e:
        logging.error(f"Error checking PDF type for {pdf_path}: {e}")
        return False

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

def extract_tables_from_pdf(pdf_path):
    """
    Extract tables from the PDF and save them as CSV files.
    
    Args:
        pdf_path (str): Path to the PDF file.
    """
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = 'output'
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
        
        if is_native_pdf(pdf_path):
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
        else:
            logging.info(f"Detected scanned PDF: {pdf_path}")
            try:
                pdf = PDF(src=pdf_path)
                ocr = TesseractOCR(
                    n_threads=1,
                    lang='eng',
                    tessdata_dir=TESSDATA_DIR  # Use the defined TESSDATA_DIR
                )
                extracted_tables = pdf.extract_tables(
                    ocr=ocr,
                    implicit_rows=True,
                    borderless_tables=True,
                    min_confidence=80
                )
                if not extracted_tables:
                    logging.warning(f"No tables found in {pdf_path}")
                else:
                    for i, table in enumerate(extracted_tables.values()):
                        df = table.df
                        if not df.empty:
                            output_file = os.path.join(output_dir, f'{base_name}_table_scanned_{i}.csv')
                            df.to_csv(output_file, index=False, encoding='utf-8')
                            logging.info(f"Saved table to {output_file}")
                        else:
                            logging.warning(f"Empty table skipped in {pdf_path}")
            except Exception as e:
                logging.error(f"Error processing scanned PDF {pdf_path}: {e}")
                import traceback
                logging.error(traceback.format_exc())
    
    finally:
        try:
            safe_remove_temp_dir(temp_dir)
        except Exception as e:
            logging.error(f"Failed to clean up temp dir {temp_dir}: {e}")

if __name__ == "__main__":
    input_dir = 'input_pdfs'  # Directory containing PDF file
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            logging.info(f"Starting extraction for {pdf_file}")
            extract_tables_from_pdf(pdf_file)
        else:
            logging.error(f"File not found: {pdf_file}")