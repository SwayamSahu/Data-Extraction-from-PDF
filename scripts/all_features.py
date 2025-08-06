import fitz  
import cv2
import numpy as np
import math
import pytesseract
from PIL import Image
import os
import re
import json
import tempfile
from paddleocr import PaddleOCR
import pypdfium2 as pdfium  # Added for PaddleOCR rendering
import traceback
import sys

# Standard page sizes (in points, at 72 DPI)
STANDARD_SIZES = {
    'A4': (595, 842),
    'Letter': (612, 792),
    'Legal': (612, 1008),
    'A3': (842, 1191)
}
DPI = 300  # Match your processing DPI
TARGET_WIDTH = int(STANDARD_SIZES['A4'][0] * (DPI / 72))
TARGET_HEIGHT = int(STANDARD_SIZES['A4'][1] * (DPI / 72))

# Import OCR engines
try:
    PADDLE_AVAILABLE = True
    print("PaddleOCR available")
except ImportError:
    PADDLE_AVAILABLE = False
    print("PaddleOCR not available. Install with: pip install paddleocr")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    print("EasyOCR imported successfully")
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR not available. Install with: pip install easyocr")

# Set up Tesseract command path
pytesseract.pytesseract.tesseract_cmd = r"C:\\Users\\SwayamPrakashSahu\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"

# Folder paths
input_folder = r'C:\Users\SwayamPrakashSahu\Downloads\AMAT_TableExtractor\input_pdfs'
output_folder = r'C:\Users\SwayamPrakashSahu\Downloads\AMAT_TableExtractor\corrected_outputs'
os.makedirs(output_folder, exist_ok=True)

# Global storage for extracted words
extracted_words_by_page = {}
special_patterns_by_page = {}
ocr_performance_by_page = {}

# Initialize OCR engines
paddle_ocr = None
easy_ocr = None

def initialize_ocr_engines():
    """Initialize all available OCR engines"""
    global paddle_ocr, easy_ocr
    
    print("Initializing OCR engines...")
    
    # Initialize PaddleOCR with new settings
    if PADDLE_AVAILABLE:
        try:
            paddle_ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False
            )
            print("✓ PaddleOCR initialized successfully")
        except Exception as e:
            print(f"✗ PaddleOCR initialization failed: {e}")
            traceback.print_exc()
            paddle_ocr = None
    
    # Initialize EasyOCR
    if EASYOCR_AVAILABLE:
        try:
            easy_ocr = easyocr.Reader(['en'], gpu=False)
            print("✓ EasyOCR initialized successfully")
        except Exception as e:
            print(f"✗ EasyOCR initialization failed: {e}")
            traceback.print_exc()
            easy_ocr = None
    
    # Check Tesseract
    try:
        pytesseract.image_to_string(np.ones((100, 100), dtype=np.uint8) * 255)
        print("✓ Tesseract is available")
    except Exception as e:
        print(f"✗ Tesseract not available: {e}")
        traceback.print_exc()

def get_rotation_angle(pil_image):
    """Detect page orientation using Tesseract OSD"""
    try:
        osd = pytesseract.image_to_osd(pil_image)
        angle = 0
        for line in osd.splitlines():
            if "Rotate" in line:
                angle = int(line.split(":")[-1].strip())
                break
        return angle
    except Exception as e:
        print(f"Orientation detection failed: {e}")
        return 0

def rotate_image(pil_image, angle):
    """Rotate PIL image while preserving content"""
    if angle == 0:
        return pil_image
    return pil_image.rotate(-angle, expand=True)

def scale_and_center_image(pil_image, target_width, target_height):
    """Standardize image size while preserving aspect ratio"""
    original_width, original_height = pil_image.size
    scale = min(target_width/original_width, target_height/original_height)
    
    if scale < 1:
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        img = pil_image.resize((new_width, new_height), Image.LANCZOS)
    else:
        img = pil_image
        new_width, new_height = original_width, original_height
    
    background = Image.new("RGB", (target_width, target_height), (255, 255, 255))
    x = (target_width - new_width) // 2
    y = (target_height - new_height) // 2
    background.paste(img, (x, y))
    
    return background

def extract_images_from_pdf(pdf_path):
    """Extract each page of PDF as high-resolution image"""
    print("Extracting images from PDF...")
    page_images = []
    pdf = fitz.open(pdf_path)

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        pix = page.get_pixmap(dpi=300)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        if pix.n == 1:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif pix.n == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            
        page_images.append((page_num, image))
        print(f"Extracted page {page_num + 1}")

    pdf.close()
    return page_images

def preprocess_for_ocr(image):
    """Preprocess image for better OCR results"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return binary

def tesseract_ocr(image):
    """Extract text using Tesseract OCR"""
    try:
        processed = preprocess_for_ocr(image)
        word_data = pytesseract.image_to_data(processed, config=r'--oem 3 --psm 3', output_type=pytesseract.Output.DICT)
        
        words_list = []
        word_details = []
        confidences = []
        
        for i in range(len(word_data['text'])):
            text = word_data['text'][i].strip()
            conf = int(word_data['conf'][i])
            
            if text and conf > 30 and re.match(r'^[a-zA-Z0-9\s\-\.\,\:\(\)\[\]\/\&\%\$\#\@\!\+\=\*]+', text):
                words_list.append(text)
                confidences.append(conf)
                word_details.append({
                    'text': text,
                    'confidence': conf,
                    'x': word_data['left'][i],
                    'y': word_data['top'][i],
                    'width': word_data['width'][i],
                    'height': word_data['height'][i],
                    'engine': 'tesseract'
                })
        
        full_text = pytesseract.image_to_string(processed, config=r'--oem 3 --psm 3')
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return {
            'engine': 'tesseract',
            'words_list': words_list,
            'word_details': word_details,
            'full_text': full_text,
            'word_count': len(words_list),
            'avg_confidence': avg_confidence,
            'success': True
        }
        
    except Exception as e:
        print(f"Tesseract OCR error: {e}")
        return {
            'engine': 'tesseract',
            'words_list': [],
            'word_details': [],
            'full_text': '',
            'word_count': 0,
            'avg_confidence': 0,
            'success': False,
            'error': str(e)
        }

def paddle_ocr_extract(image):
    """Extract text using PaddleOCR with new rendering method"""
    if not paddle_ocr:
        return {
            'engine': 'paddleocr',
            'words_list': [],
            'word_details': [],
            'full_text': '',
            'word_count': 0,
            'avg_confidence': 0,
            'success': False,
            'error': 'PaddleOCR not available'
        }
    
    try:
        # Convert image to RGB format
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Run OCR prediction using correct API method
        result = paddle_ocr.predict(image_rgb)
        
        # Initialize result containers
        words_list = []
        word_details = []
        confidences = []
        full_text_parts = []

        # Process results
        if result and len(result) > 0:
            page_result = result[0]  # First page results
            
            for line in page_result:
                if line is None:
                    continue
                    
                # Extract bounding box and text info
                bbox = line[0]
                text_info = line[1]
                
                if text_info is None:
                    continue
                    
                text = text_info[0]
                confidence = float(text_info[1]) * 100
                
                # Skip low confidence or empty text
                if not text or confidence < 30:
                    continue
                
                # Calculate bounding box dimensions
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x = int(min(x_coords))
                y = int(min(y_coords))
                width = int(max(x_coords) - min(x_coords))
                height = int(max(y_coords) - min(y_coords))
                
                # Split text into individual words
                words = text.split()
                for word in words:
                    if word.strip():
                        words_list.append(word.strip())
                        confidences.append(confidence)
                        word_details.append({
                            'text': word.strip(),
                            'confidence': confidence,
                            'x': x,
                            'y': y,
                            'width': width // max(len(words), 1),
                            'height': height,
                            'engine': 'paddleocr'
                        })

                full_text_parts.append(text)
            
        # Prepare final result
        full_text = ' '.join(full_text_parts)
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return {
            'engine': 'paddleocr',
            'words_list': words_list,
            'word_details': word_details,
            'full_text': full_text,
            'word_count': len(words_list),
            'avg_confidence': avg_confidence,
            'success': True
        }
        
    except Exception as e:
        print(f"PaddleOCR error: {e}")
        return {
            'engine': 'paddleocr',
            'words_list': [],
            'word_details': [],
            'full_text': '',
            'word_count': 0,
            'avg_confidence': 0,
            'success': False,
            'error': str(e)
        }

def easyocr_extract(image):
    """Extract text using EasyOCR"""
    if not easy_ocr:
        return {
            'engine': 'easyocr',
            'words_list': [],
            'word_details': [],
            'full_text': '',
            'word_count': 0,
            'avg_confidence': 0,
            'success': False,
            'error': 'EasyOCR not available'
        }
    
    try:
        results = easy_ocr.readtext(image)
        
        words_list = []
        word_details = []
        confidences = []
        full_text_parts = []
        
        for (bbox, text, confidence) in results:
            text = text.strip()
            confidence_pct = confidence * 100
            
            if text and confidence_pct > 30:
                # Calculate bounding box
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x = int(min(x_coords))
                y = int(min(y_coords))
                width = int(max(x_coords) - min(x_coords))
                height = int(max(y_coords) - min(y_coords))
                
                # Split text into individual words
                individual_words = text.split()
                for word in individual_words:
                    if word.strip():
                        words_list.append(word.strip())
                        confidences.append(confidence_pct)
                        word_details.append({
                            'text': word.strip(),
                            'confidence': confidence_pct,
                            'x': x,
                            'y': y,
                            'width': width // len(individual_words),
                            'height': height,
                            'engine': 'easyocr'
                        })
                
                full_text_parts.append(text)
        
        full_text = ' '.join(full_text_parts)
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return {
            'engine': 'easyocr',
            'words_list': words_list,
            'word_details': word_details,
            'full_text': full_text,
            'word_count': len(words_list),
            'avg_confidence': avg_confidence,
            'success': True
        }
        
    except Exception as e:
        print(f"EasyOCR error: {e}")
        return {
            'engine': 'easyocr',
            'words_list': [],
            'word_details': [],
            'full_text': '',
            'word_count': 0,
            'avg_confidence': 0,
            'success': False,
            'error': str(e)
        }

def multi_ocr_extract(image, page_num):
    """
    Run all available OCR engines and select the best result
    Best result is determined by highest word count with reasonable confidence
    """
    print(f"Running multi-OCR analysis for page {page_num + 1}...")
    
    ocr_results = []
    
    # Run Tesseract
    print("  Running Tesseract OCR...")
    tesseract_result = tesseract_ocr(image)
    ocr_results.append(tesseract_result)
    
    # Run PaddleOCR
    if PADDLE_AVAILABLE and paddle_ocr:
        print("  Running PaddleOCR...")
        paddle_result = paddle_ocr_extract(image)
        ocr_results.append(paddle_result)
    
    # Run EasyOCR
    if EASYOCR_AVAILABLE and easy_ocr:
        print("  Running EasyOCR...")
        easy_result = easyocr_extract(image)
        ocr_results.append(easy_result)
    
    # Evaluate results and select the best one
    best_result = None
    best_score = 0
    
    print("  OCR Results Comparison:")
    for result in ocr_results:
        if result['success']:
            score = result['word_count'] * (result['avg_confidence'] / 100)
            
            print(f"    {result['engine']}: {result['word_count']} words, "
                  f"{result['avg_confidence']:.1f}% confidence, score: {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_result = result
        else:
            error_msg = result.get('error', 'Unknown error')
            print(f"    {result['engine']}: FAILED - {error_msg}")
    
    if best_result:
        print(f"  ✓ Best result: {best_result['engine']} "
              f"({best_result['word_count']} words, {best_result['avg_confidence']:.1f}% confidence)")
        
        # Store performance data
        ocr_performance_by_page[page_num + 1] = {
            'all_results': ocr_results,
            'best_engine': best_result['engine'],
            'best_score': best_score
        }
        
        return best_result
    else:
        print("  ✗ All OCR engines failed!")
        ocr_performance_by_page[page_num + 1] = {
            'all_results': ocr_results,
            'best_engine': 'none',
            'best_score': 0
        }
        
        return {
            'engine': 'none',
            'words_list': [],
            'word_details': [],
            'full_text': '',
            'word_count': 0,
            'avg_confidence': 0,
            'success': False
        }

def identify_special_patterns(text_list):
    """
    Identify special patterns in extracted text using regular expressions
    """
    patterns = {
        'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone_numbers': r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
        'dates': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',
        'currency': r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|INR|EUR|GBP)\b',
        'tracking_numbers': r'\b[A-Z]{2,3}\d{8,15}\b|\b\d{10,20}\b',
        'invoice_numbers': r'\b(?:INV|INVOICE|REF|NO)[#:\-\s]*[A-Z0-9]{3,15}\b',
        'po_numbers': r'\b(?:PO|P\.O\.|PURCHASE\s+ORDER)[#:\-\s]*[A-Z0-9]{3,15}\b',
        'percentages': r'\b\d{1,3}(?:\.\d{1,2})?%\b',
        'quantities': r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:KG|LBS|PCS|UNITS|QTY|PIECES|TONS|BOXES)\b',
        'addresses': r'\b\d+\s+[A-Za-z\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd)\b',
        'postal_codes': r'\b\d{5,6}(?:[-\s]\d{4})?\b',
        'reference_codes': r'\b[A-Z]{2,4}[-_]?\d{3,10}[-_]?[A-Z0-9]*\b',
        'weights': r'\b\d+(?:\.\d+)?\s*(?:KG|LB|LBS|POUND|POUNDS|TON|TONS|GRAM|GRAMS|OZ)\b',
        'dimensions': r'\b\d+(?:\.\d+)?\s*(?:X|x)\s*\d+(?:\.\d+)?(?:\s*(?:X|x)\s*\d+(?:\.\d+)?)?\s*(?:CM|MM|IN|INCH|INCHES|FT|FEET|M|METER|METERS)\b'
    }
    
    found_patterns = {}
    combined_text = ' '.join(text_list)
    
    for pattern_name, pattern_regex in patterns.items():
        matches = re.findall(pattern_regex, combined_text, re.IGNORECASE)
        if matches:
            # Flatten possible tuple matches
            flat_matches = [m[0] if isinstance(m, tuple) and len(m) > 0 else m for m in matches]
            unique_matches = list(dict.fromkeys(flat_matches))
            found_patterns[pattern_name] = unique_matches
    
    return found_patterns

def extract_and_store_words(image, page_num):
    """
    Extract words from image using multi-OCR and store them page-wise
    Also identify special patterns
    """
    global extracted_words_by_page, special_patterns_by_page
    
    try:
        best_result = multi_ocr_extract(image, page_num)
        
        if best_result['success']:
            extracted_words_by_page[page_num + 1] = {
                'full_text': best_result['full_text'],
                'words_list': best_result['words_list'],
                'word_details': best_result['word_details'],
                'word_count': best_result['word_count'],
                'avg_confidence': best_result['avg_confidence'],
                'ocr_engine': best_result['engine']
            }
            
            special_patterns = identify_special_patterns(best_result['words_list'])
            special_patterns_by_page[page_num + 1] = special_patterns
            
            print(f"Page {page_num + 1}: Extracted {best_result['word_count']} words using {best_result['engine']}")
            if special_patterns:
                print(f"  Found special patterns: {list(special_patterns.keys())}")
            
            return best_result['words_list'], best_result['word_details'], special_patterns
        else:
            extracted_words_by_page[page_num + 1] = {
                'full_text': '',
                'words_list': [],
                'word_details': [],
                'word_count': 0,
                'avg_confidence': 0,
                'ocr_engine': 'none'
            }
            special_patterns_by_page[page_num + 1] = {}
            return [], [], {}
        
    except Exception as e:
        print(f"Word extraction error for page {page_num + 1}: {e}")
        extracted_words_by_page[page_num + 1] = {
            'full_text': '',
            'words_list': [],
            'word_details': [],
            'word_count': 0,
            'avg_confidence': 0,
            'ocr_engine': 'error'
        }
        special_patterns_by_page[page_num + 1] = {}
        return [], [], {}

def get_detailed_ocr_data(image):
    """Get detailed OCR data using the best available OCR engine"""
    best_result = multi_ocr_extract(image, -1)
    
    if best_result['success'] and best_result['word_details']:
        words = []
        for word_detail in best_result['word_details']:
            words.append({
                'text': word_detail['text'],
                'confidence': word_detail['confidence'],
                'x': word_detail['x'],
                'y': word_detail['y'],
                'w': word_detail['width'],
                'h': word_detail['height']
            })
        
        return {
            'words': words,
            'score': best_result['word_count'] * (best_result['avg_confidence'] / 100),
            'word_count': best_result['word_count'],
            'avg_confidence': best_result['avg_confidence']
        }
    else:
        return {'words': [], 'score': 0, 'word_count': 0, 'avg_confidence': 0}

def detect_table_structure(words):
    """
    Detect table structure from OCR word positions
    """
    if not words or len(words) < 3:
        return {'rows': [], 'columns': [], 'is_table': False, 'alignment_score': 0}
    
    y_positions = [word['y'] + word['h']//2 for word in words]
    y_tolerance = np.median([word['h'] for word in words]) * 0.5
    
    y_positions_sorted = sorted(set(y_positions))
    rows = []
    current_row_y = y_positions_sorted[0]
    current_row = [current_row_y]
    
    for y in y_positions_sorted[1:]:
        if abs(y - current_row_y) <= y_tolerance:
            current_row.append(y)
        else:
            if current_row:
                rows.append(np.mean(current_row))
            current_row = [y]
            current_row_y = y
    
    if current_row:
        rows.append(np.mean(current_row))
    
    x_positions = [word['x'] + word['w']//2 for word in words]
    x_tolerance = np.median([word['w'] for word in words]) * 0.3
    
    x_positions_sorted = sorted(set(x_positions))
    columns = []
    current_col_x = x_positions_sorted[0]
    current_col = [current_col_x]
    
    for x in x_positions_sorted[1:]:
        if abs(x - current_col_x) <= x_tolerance:
            current_col.append(x)
        else:
            if current_col:
                columns.append(np.mean(current_col))
            current_col = [x]
            current_col_x = x
    
    if current_col:
        columns.append(np.mean(current_col))
    
    is_table = len(rows) >= 2 and len(columns) >= 2
    
    if is_table:
        row_alignment_score = 0
        col_alignment_score = 0
        
        for word in words:
            word_y = word['y'] + word['h']//2
            word_x = word['x'] + word['w']//2
            
            closest_row = min(rows, key=lambda r: abs(r - word_y))
            closest_col = min(columns, key=lambda c: abs(c - word_x))
            
            if abs(closest_row - word_y) <= y_tolerance:
                row_alignment_score += 1
            if abs(closest_col - word_x) <= x_tolerance:
                col_alignment_score += 1
        
        total_words = len(words)
        alignment_score = (row_alignment_score + col_alignment_score) / (2 * total_words) if total_words > 0 else 0
    else:
        alignment_score = 0
    
    return {
        'rows': rows,
        'columns': columns,
        'is_table': is_table,
        'alignment_score': alignment_score,
        'row_count': len(rows),
        'col_count': len(columns)
    }

def detect_text_angles_from_structure(words, table_info):
    """Detect text angles from table structure and word alignment"""
    if not words:
        return []
    
    angles = []
    
    if table_info['is_table'] and table_info['alignment_score'] > 0.5:
        rows = table_info['rows']
        row_tolerance = np.median([word['h'] for word in words]) * 0.5
        
        for row_y in rows:
            row_words = []
            for word in words:
                word_y = word['y'] + word['h']//2
                if abs(word_y - row_y) <= row_tolerance:
                    row_words.append(word)
            
            if len(row_words) >= 2:
                row_words.sort(key=lambda w: w['x'])
                
                first_word = row_words[0]
                last_word = row_words[-1]
                
                y1 = first_word['y'] + first_word['h']//2
                x1 = first_word['x'] + first_word['w']//2
                y2 = last_word['y'] + last_word['h']//2
                x2 = last_word['x'] + last_word['w']//2
                
                if abs(x2 - x1) > 10:
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    
                    if angle > 45:
                        angle -= 90
                    elif angle < -45:
                        angle += 90
                    
                    if abs(angle) <= 15:
                        angles.append(angle)
    
    word_centers = [(word['x'] + word['w']//2, word['y'] + word['h']//2) for word in words]
    
    if len(word_centers) >= 3:
        angles_sample = []
        
        for i in range(0, len(word_centers)-1, 2):
            for j in range(i+1, min(i+5, len(word_centers))):
                x1, y1 = word_centers[i]
                x2, y2 = word_centers[j]
                
                if abs(x2 - x1) > 20:
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    
                    if angle > 45:
                        angle -= 90
                    elif angle < -45:
                        angle += 90
                    
                    if abs(angle) <= 15:
                        angles_sample.append(angle)
        
        angles.extend(angles_sample)
    
    return angles

def enhanced_get_ocr_score(image, page_num=None):
    """Enhanced OCR scoring that considers table structure and extracts words"""
    ocr_data = get_detailed_ocr_data(image)
    
    if page_num is not None:
        extract_and_store_words(image, page_num)
    
    if not ocr_data['words']:
        return 0, 0, {'rows': [], 'columns': [], 'is_table': False, 'alignment_score': 0}
    
    table_info = detect_table_structure(ocr_data['words'])
    
    base_score = ocr_data['score']
    
    if table_info['is_table']:
        structure_bonus = table_info['alignment_score'] * 50
        base_score += structure_bonus
    
    high_conf_words = sum(1 for word in ocr_data['words'] if word['confidence'] > 80)
    confidence_bonus = high_conf_words * 0.5
    
    total_score = base_score + confidence_bonus
    
    return total_score, len(ocr_data['words']), table_info

def detect_skew_comprehensive(image, page_num=None):
    """
    Comprehensive skew detection using multiple methods
    """
    print(f"Detecting skew for page {page_num + 1 if page_num is not None else 'N/A'}...")
    
    skew_angles = []
    
    # Method 1: Hough Line Transform
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for i in range(min(20, len(lines))):
                rho, theta = lines[i][0]
                angle = math.degrees(theta) - 90
                if abs(angle) <= 45:
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                skew_angles.append(('hough_lines', median_angle))
                print(f"  Hough Lines: {median_angle:.2f}°")
        else:
            print("  Hough Lines: No lines detected")
    
    except Exception as e:
        print(f"  Hough Lines failed: {e}")
    
    # Method 2: OCR-based structure analysis
    try:
        score, word_count, table_info = enhanced_get_ocr_score(image, page_num)
        
        if word_count > 5:
            ocr_data = get_detailed_ocr_data(image)
            if ocr_data['words']:
                text_angles = detect_text_angles_from_structure(ocr_data['words'], table_info)
                
                if text_angles:
                    avg_text_angle = np.median(text_angles)
                    skew_angles.append(('ocr_structure', avg_text_angle))
                    print(f"  OCR Structure: {avg_text_angle:.2f}°")
    
    except Exception as e:
        print(f"  OCR Structure analysis failed: {e}")
    
    # Method 3: Projection Profile
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        test_angles = np.arange(-10, 11, 0.5)
        best_angle = 0
        best_variance = 0
        
        for angle in test_angles:
            h, w = gray.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            projection = np.sum(rotated, axis=1)
            variance = np.var(projection)
            
            if variance > best_variance:
                best_variance = variance
                best_angle = angle
        
        if abs(best_angle) <= 15:
            skew_angles.append(('projection_profile', best_angle))
            print(f"  Projection Profile: {best_angle:.2f}°")
    
    except Exception as e:
        print(f"  Projection Profile failed: {e}")
    
    # Method 4: Contour-based detection
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        large_contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        if large_contours:
            angles = []
            for contour in large_contours[:10]:
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                
                if angle < -45:
                    angle += 90
                elif angle > 45:
                    angle -= 90
                
                if abs(angle) <= 15:
                    angles.append(angle)
            
            if angles:
                contour_angle = np.median(angles)
                skew_angles.append(('contours', contour_angle))
                print(f"  Contours: {contour_angle:.2f}°")
    
    except Exception as e:
        print(f"  Contour detection failed: {e}")
    
    # Determine final skew angle
    if skew_angles:
        weights = {
            'ocr_structure': 0.4,
            'hough_lines': 0.3,
            'projection_profile': 0.2,
            'contours': 0.1
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for method, angle in skew_angles:
            weight = weights.get(method, 0.1)
            weighted_sum += angle * weight
            total_weight += weight
        
        final_angle = weighted_sum / total_weight if total_weight > 0 else 0
        
        print(f"  Final skew angle: {final_angle:.2f}°")
        return final_angle
    else:
        print("  No skew detected")
        return 0

def correct_skew_and_enhance(image, skew_angle):
    """
    Correct skew and enhance image quality
    """
    if abs(skew_angle) < 0.1:
        print("  No rotation needed")
        return image
    
    print(f"  Rotating by {skew_angle:.2f}°")
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    
    cos_val = abs(M[0, 0])
    sin_val = abs(M[0, 1])
    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    corrected = cv2.warpAffine(image, M, (new_w, new_h), 
                              flags=cv2.INTER_CUBIC, 
                              borderMode=cv2.BORDER_CONSTANT, 
                              borderValue=(255, 255, 255))
    
    return corrected

def process_pdf(input_pdf_path, output_pdf_path):
    """Process a single PDF file"""
    global extracted_words_by_page, special_patterns_by_page, ocr_performance_by_page
    
    extracted_words_by_page = {}
    special_patterns_by_page = {}
    ocr_performance_by_page = {}
    
    print("=" * 60)
    print(f"PROCESSING: {os.path.basename(input_pdf_path)}")
    print("=" * 60)

    initialize_ocr_engines()
    
    if not os.path.exists(input_pdf_path):
        print(f"Error: PDF file not found at {input_pdf_path}")
        return
    
    try:
        page_images = extract_images_from_pdf(input_pdf_path)
        
        if not page_images:
            print("No pages found in PDF")
            return
        
        print(f"\nProcessing {len(page_images)} pages...")
        
        corrected_images = []
        
        for page_num, original_image in page_images:
            print(f"\n--- Processing Page {page_num + 1} ---")
        
            pil_img = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            
            major_angle = get_rotation_angle(pil_img)
            if major_angle:
                print(f"  Correcting orientation: {major_angle}°")
            oriented_img = rotate_image(pil_img, major_angle)
            oriented_cv = cv2.cvtColor(np.array(oriented_img), cv2.COLOR_RGB2BGR)
            
            skew_angle = detect_skew_comprehensive(oriented_cv, page_num)
            deskewed_img = correct_skew_and_enhance(oriented_cv, skew_angle)
            
            print("  Final OCR extraction...")
            extract_and_store_words(deskewed_img, page_num)
            
            final_pil = Image.fromarray(cv2.cvtColor(deskewed_img, cv2.COLOR_BGR2RGB))
            standardized = scale_and_center_image(final_pil, TARGET_WIDTH, TARGET_HEIGHT)
            corrected_images.append(standardized)
        
        save_images_to_pdf(corrected_images, output_pdf_path)
        print_comprehensive_summary(output_pdf_path)
        analyze_document_structure()
        
        print(f"\n✅ Processing completed for: {os.path.basename(input_pdf_path)}")
        print(f"   Output saved to: {output_pdf_path}")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        traceback.print_exc()

def save_images_to_pdf(images, output_pdf_path):
    """Save PIL images to PDF with proper DPI"""
    print(f"\nSaving corrected PDF to: {output_pdf_path}")
    try:
        if images:
            images[0].save(
                output_pdf_path,
                save_all=True,
                append_images=images[1:],
                format='PDF',
                dpi=(DPI, DPI),
                quality=100
            )
            print(f"✓ Saved {len(images)} pages")
        else:
            print("✗ No images to save")
    except Exception as e:
        print(f"Error saving PDF: {e}")
        traceback.print_exc()

def print_comprehensive_summary(output_pdf_path):
    """
    Print comprehensive summary of OCR results and patterns found
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE OCR ANALYSIS SUMMARY")
    print("=" * 80)
    
    total_words = 0
    total_pages = len(extracted_words_by_page)
    
    if total_pages == 0:
        print("No OCR data available for summary")
        return
    
    print(f"\n📊 PAGE-WISE ANALYSIS ({total_pages} pages processed):")
    print("-" * 50)
    
    for page_num in sorted(extracted_words_by_page.keys()):
        page_data = extracted_words_by_page[page_num]
        words_count = page_data['word_count']
        confidence = page_data['avg_confidence']
        engine = page_data['ocr_engine']
        total_words += words_count
        
        print(f"Page {page_num}: {words_count} words | {confidence:.1f}% confidence | Engine: {engine}")
        
        if page_num in special_patterns_by_page and special_patterns_by_page[page_num]:
            patterns = special_patterns_by_page[page_num]
            pattern_summary = []
            for pattern_type, matches in patterns.items():
                pattern_summary.append(f"{pattern_type}({len(matches)})")
            print(f"         Patterns: {', '.join(pattern_summary)}")
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"   Total words extracted: {total_words}")
    print(f"   Average words per page: {total_words / total_pages:.1f}")
    
    print(f"\n🔧 OCR ENGINE PERFORMANCE:")
    print("-" * 30)
    
    engine_stats = {}
    for page_num, page_data in extracted_words_by_page.items():
        engine = page_data['ocr_engine']
        if engine not in engine_stats:
            engine_stats[engine] = {'pages': 0, 'total_words': 0, 'total_confidence': 0}
        
        engine_stats[engine]['pages'] += 1
        engine_stats[engine]['total_words'] += page_data['word_count']
        engine_stats[engine]['total_confidence'] += page_data['avg_confidence']
    
    for engine, stats in engine_stats.items():
        avg_confidence = stats['total_confidence'] / stats['pages'] if stats['pages'] > 0 else 0
        avg_words = stats['total_words'] / stats['pages'] if stats['pages'] > 0 else 0
        print(f"   {engine}: {stats['pages']} pages | Avg: {avg_words:.1f} words | {avg_confidence:.1f}% confidence")
    
    print(f"\n🎯 SPECIAL PATTERNS DETECTED:")
    print("-" * 35)
    
    all_patterns = {}
    for page_patterns in special_patterns_by_page.values():
        for pattern_type, matches in page_patterns.items():
            if pattern_type not in all_patterns:
                all_patterns[pattern_type] = []
            all_patterns[pattern_type].extend(matches)
    
    if all_patterns:
        for pattern_type, matches in all_patterns.items():
            unique_matches = list(dict.fromkeys(matches))
            print(f"   {pattern_type.replace('_', ' ').title()}: {len(unique_matches)} unique")
            
            if unique_matches:
                examples = unique_matches[:3]
                print(f"      Examples: {', '.join(str(ex) for ex in examples)}")
    else:
        print("   No special patterns detected")
    
    if ocr_performance_by_page:
        print(f"\n🔍 DETAILED OCR COMPARISON:")
        print("-" * 40)
        
        for page_num in sorted(ocr_performance_by_page.keys()):
            perf_data = ocr_performance_by_page[page_num]
            print(f"\nPage {page_num} - Best: {perf_data['best_engine']} (score: {perf_data['best_score']:.2f})")
            
            for result in perf_data['all_results']:
                if result['success']:
                    score = result['word_count'] * (result['avg_confidence'] / 100)
                    status = "✓" if result['engine'] == perf_data['best_engine'] else " "
                    print(f"  {status} {result['engine']}: {result['word_count']} words, {result['avg_confidence']:.1f}% conf, score: {score:.2f}")
                else:
                    print(f"    {result['engine']}: FAILED")
    
    print(f"\n💾 OUTPUT:")
    print(f"   Corrected PDF saved to: {output_pdf_path}")
    
    save_detailed_results(output_pdf_path)
    
    print("=" * 80)

def save_detailed_results(output_pdf_path):
    """
    Save detailed OCR results to JSON file for further analysis
    """
    try:
        results_data = {
            'processing_summary': {
                'total_pages': len(extracted_words_by_page),
                'total_words': sum(page['word_count'] for page in extracted_words_by_page.values()),
                'processing_date': str(np.datetime64('now')),
                'input_file': input_folder,
                'output_file': output_folder
            },
            'page_wise_results': extracted_words_by_page,
            'special_patterns': special_patterns_by_page,
            'ocr_performance': ocr_performance_by_page
        }
        
        json_output_file = output_pdf_path.replace('.pdf', '_analysis.json')
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"   Detailed analysis saved to: {json_output_file}")
        
    except Exception as e:
        print(f"   Warning: Could not save detailed results: {e}")

def analyze_document_structure():
    """
    Analyze overall document structure and provide insights
    """
    print(f"\n📋 DOCUMENT STRUCTURE ANALYSIS:")
    print("-" * 40)
    
    if not extracted_words_by_page:
        print("No data available for analysis")
        return
    
    word_counts = [page['word_count'] for page in extracted_words_by_page.values()]
    
    print(f"Word count statistics:")
    print(f"   Min: {min(word_counts)} words")
    print(f"   Max: {max(word_counts)} words")
    print(f"   Mean: {np.mean(word_counts):.1f} words")
    print(f"   Median: {np.median(word_counts):.1f} words")
    print(f"   Std Dev: {np.std(word_counts):.1f}")
    
    avg_words = np.mean(word_counts)
    threshold = avg_words * 0.3
    
    sparse_pages = [page_num for page_num, page_data in extracted_words_by_page.items() 
                   if page_data['word_count'] < threshold]
    
    if sparse_pages:
        print(f"   Potential header/footer pages: {sparse_pages}")
    
    confidences = [page['avg_confidence'] for page in extracted_words_by_page.values()]
    
    print(f"\nConfidence statistics:")
    print(f"   Average confidence: {np.mean(confidences):.1f}%")
    print(f"   Confidence range: {min(confidences):.1f}% - {max(confidences):.1f}%")
    
    low_conf_threshold = 70
    low_conf_pages = [page_num for page_num, page_data in extracted_words_by_page.items() 
                     if page_data['avg_confidence'] < low_conf_threshold]
    
    if low_conf_pages:
        print(f"   Pages with low confidence (<{low_conf_threshold}%): {low_conf_pages}")

# Main execution
if __name__ == "__main__":
    print("Starting Multi-OCR PDF Processing...")
    print(f"Input Folder: {input_folder}")
    print(f"Output Folder: {output_folder}")

    pdf_files = [f for f in os.listdir(input_folder) 
                if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(input_folder, f))]
    
    if not pdf_files:
        print("No PDF files found in input folder")
        exit()
    
    print(f"Found {len(pdf_files)} PDFs to process")

    for pdf_file in pdf_files:
        input_path = os.path.join(input_folder, pdf_file)
        output_path = os.path.join(output_folder, f"corrected_{pdf_file}")
        
        process_pdf(input_path, output_path)
        print("\n" + "=" * 80 + "\n")
    
    print(f"\n✅ Processing completed successfully!")
    print(f"✅ Batch processing completed! Processed {len(pdf_files)} files")
    print(f"   Check output folder: {output_folder}")
