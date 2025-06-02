#  PDF : Skew Angle,Rotation and OCR - Complete Fix - Enhanced for All Table Types
# With Word Extraction, Pattern Recognition and Multi-OCR Engine Support
# Supports: Tesseract, PaddleOCR, and EasyOCR with automatic best selection

import fitz  
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pytesseract
from PIL import Image
import os
import re
from collections import Counter
import json

# Import OCR engines
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
    print("PaddleOCR imported successfully")
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

# Step 1: Set up Tesseract command path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Step 2: Update file path
pdf_file = r'C:\Users\hp\Downloads\amat_final_work\Data-Extraction-from-PDF\A1.pdf'
output_file = r'C:\Users\hp\Downloads\amat_final_work\Data-Extraction-from-PDF\corrected4_output.pdf'

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
    
    # Initialize PaddleOCR
    if PADDLE_AVAILABLE:
        try:
            # Initialize PaddleOCR with updated parameters
            paddle_ocr = PaddleOCR(
                use_textline_orientation=True,  # Updated parameter name
                lang='en',                      # Set language to English
                # show_log=False                  # Reduce verbose logging
                # Removed use_gpu parameter as it's not supported in newer versions
            )
            print("✓ PaddleOCR initialized successfully")
        except Exception as e:
            print(f"✗ PaddleOCR initialization failed: {e}")
            paddle_ocr = None
    
    # Initialize EasyOCR
    if EASYOCR_AVAILABLE:
        try:
            easy_ocr = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have CUDA
            print("✓ EasyOCR initialized successfully")
        except Exception as e:
            print(f"✗ EasyOCR initialization failed: {e}")
            easy_ocr = None
    
    # Check Tesseract
    try:
        pytesseract.image_to_string(np.ones((100, 100), dtype=np.uint8) * 255)
        print("✓ Tesseract is available")
    except Exception as e:
        print(f"✗ Tesseract not available: {e}")

def extract_images_from_pdf(pdf_path):
    """Extract each page of PDF as high-resolution image"""
    print("Extracting images from PDF...")
    page_images = []
    pdf = fitz.open(pdf_path)

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        # High resolution for better OCR
        pix = page.get_pixmap(dpi=300)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Convert to BGR format
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
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply OTSU thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return binary

def tesseract_ocr(image):
    """Extract text using Tesseract OCR"""
    try:
        processed = preprocess_for_ocr(image)
        
        # Get word-level data
        word_data = pytesseract.image_to_data(processed, config=r'--oem 3 --psm 3', output_type=pytesseract.Output.DICT)
        
        words_list = []
        word_details = []
        confidences = []
        
        for i in range(len(word_data['text'])):
            text = word_data['text'][i].strip()
            conf = int(word_data['conf'][i])
            
            if (text and len(text) > 0 and conf > 30 and 
                re.match(r'^[a-zA-Z0-9\s\-\.\,\:\(\)\[\]\/\&\%\$\#\@\!\+\=\*]+', text)):
                
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
        
        # Get full text
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

# def paddle_ocr_extract(image):
#     """Extract text using PaddleOCR"""
#     if not paddle_ocr:
#         return {
#             'engine': 'paddleocr',
#             'words_list': [],
#             'word_details': [],
#             'full_text': '',
#             'word_count': 0,
#             'avg_confidence': 0,
#             'success': False,
#             'error': 'PaddleOCR not available'
#         }
    
#     try:
#         # PaddleOCR works with PIL Image or numpy array
#         result = paddle_ocr.predict(image)

#         # Validate result structure
#         if not result or not isinstance(result, list) or len(result) == 0 or not isinstance(result[0], list):
#             raise ValueError("Invalid result format from PaddleOCR")
        
#         words_list = []
#         word_details = []
#         confidences = []
#         full_text_parts = []
        
#         if result and result[0]:
#             for line in result[0]:
#                 if line:
#                     bbox = line[0]  # Bounding box coordinates
#                     text_info = line[1]  # (text, confidence)
#                     text = text_info[0].strip()
#                     confidence = float(text_info[1]) * 100  # Convert to percentage
                    
#                     if (text and len(text) > 0 and confidence > 30 and 
#                         re.match(r'^[a-zA-Z0-9\s\-\.\,\:\(\)\[\]\/\&\%\$\#\@\!\+\=\*]+', text)):
                        
#                         # Calculate bounding box
#                         x_coords = [point[0] for point in bbox]
#                         y_coords = [point[1] for point in bbox]
#                         x = int(min(x_coords))
#                         y = int(min(y_coords))
#                         width = int(max(x_coords) - min(x_coords))
#                         height = int(max(y_coords) - min(y_coords))
                        
#                         # Split text into individual words
#                         individual_words = text.split()
#                         for word in individual_words:
#                             if word.strip():
#                                 words_list.append(word.strip())
#                                 confidences.append(confidence)
#                                 word_details.append({
#                                     'text': word.strip(),
#                                     'confidence': confidence,
#                                     'x': x,
#                                     'y': y,
#                                     'width': width // len(individual_words),
#                                     'height': height,
#                                     'engine': 'paddleocr'
#                                 })
                        
#                         full_text_parts.append(text)
        
#         full_text = ' '.join(full_text_parts)
#         avg_confidence = np.mean(confidences) if confidences else 0
        
#         return {
#             'engine': 'paddleocr',
#             'words_list': words_list,
#             'word_details': word_details,
#             'full_text': full_text,
#             'word_count': len(words_list),
#             'avg_confidence': avg_confidence,
#             'success': True
#         }
        
#     except Exception as e:
#         print(f"PaddleOCR error: {e}")
#         return {
#             'engine': 'paddleocr',
#             'words_list': [],
#             'word_details': [],
#             'full_text': '',
#             'word_count': 0,
#             'avg_confidence': 0,
#             'success': False,
#             'error': str(e)
#         }
def paddle_ocr_extract(image):
    """Extract text using PaddleOCR"""
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
        # Convert BGR to RGB if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Input must be numpy.ndarray")

        # # Run OCR prediction - PaddleOCR.ocr() method should be used instead of predict()
        # result = paddle_ocr.ocr(image)

        # Run OCR prediction - Use alternative parameter structure
        try:
            result = paddle_ocr.predict(image, use_angle_cls=True)
        except:
            # Fallback without angle classification
            result = paddle_ocr.predict(image)

        # Initialize result containers
        words_list = []
        word_details = []
        confidences = []
        full_text_parts = []

        # Process results - PaddleOCR returns nested list structure
        # result is a list containing one list of detected text regions for single image
        if result and len(result) > 0 and result[0] is not None:
            for line in result[0]:  # result[0] contains the list of detected text regions
                try:
                    # PaddleOCR result format: [bbox, (text, confidence)]
                    if line and len(line) >= 2:
                        bbox = line[0]  # Bounding box coordinates
                        text_info = line[1]  # (text, confidence) tuple
                        
                        if text_info and len(text_info) >= 2:
                            text = text_info[0]
                            confidence = float(text_info[1]) * 100  # Convert to percentage
                            
                            # Filter out low confidence and invalid text
                            if (text and len(text.strip()) > 0 and confidence > 30 and 
                                re.match(r'^[a-zA-Z0-9\s\-\.\,\:\(\)\[\]\/\&\%\$\#\@\!\+\=\*]+', text)):
                                
                                # Calculate bounding box dimensions
                                if bbox and len(bbox) >= 4:
                                    x_coords = [point[0] for point in bbox]
                                    y_coords = [point[1] for point in bbox]
                                    x = int(min(x_coords))
                                    y = int(min(y_coords))
                                    width = int(max(x_coords) - min(x_coords))
                                    height = int(max(y_coords) - min(y_coords))
                                else:
                                    # Default values if bbox is invalid
                                    x, y, width, height = 0, 0, 100, 20
                                
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
                            
                except Exception as e:
                    print(f"Warning: Error processing text region in PaddleOCR: {e}")
                    continue

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
        # EasyOCR readtext returns list of (bbox, text, confidence)
        results = easy_ocr.readtext(image)
        
        words_list = []
        word_details = []
        confidences = []
        full_text_parts = []
        
        for (bbox, text, confidence) in results:
            text = text.strip()
            confidence_pct = confidence * 100  # Convert to percentage
            
            if (text and len(text) > 0 and confidence_pct > 30 and 
                re.match(r'^[a-zA-Z0-9\s\-\.\,\:\(\)\[\]\/\&\%\$\#\@\!\+\=\*]+', text)):
                
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
            # Calculate score: word_count * (avg_confidence/100)
            # This balances quantity and quality
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
        
        # Return empty result
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
    Returns dictionary with pattern types and matches
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
            # Remove duplicates while preserving order
            unique_matches = list(dict.fromkeys(matches))
            found_patterns[pattern_name] = unique_matches
    
    return found_patterns

def extract_and_store_words(image, page_num):
    """
    Extract words from image using multi-OCR and store them page-wise
    Also identify special patterns
    """
    global extracted_words_by_page, special_patterns_by_page
    
    try:
        # Use multi-OCR extraction
        best_result = multi_ocr_extract(image, page_num)
        
        if best_result['success']:
            # Store words for this page
            extracted_words_by_page[page_num + 1] = {
                'full_text': best_result['full_text'],
                'words_list': best_result['words_list'],
                'word_details': best_result['word_details'],
                'word_count': best_result['word_count'],
                'avg_confidence': best_result['avg_confidence'],
                'ocr_engine': best_result['engine']
            }
            
            # Identify special patterns
            special_patterns = identify_special_patterns(best_result['words_list'])
            special_patterns_by_page[page_num + 1] = special_patterns
            
            print(f"Page {page_num + 1}: Extracted {best_result['word_count']} words using {best_result['engine']}")
            if special_patterns:
                print(f"  Found special patterns: {list(special_patterns.keys())}")
            
            return best_result['words_list'], best_result['word_details'], special_patterns
        else:
            # Store empty results
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
    # Use multi-OCR to get the best result
    best_result = multi_ocr_extract(image, -1)  # Use -1 as temp page number
    
    if best_result['success'] and best_result['word_details']:
        # Convert to the expected format
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
    
    # Group words by approximate Y positions (rows)
    y_positions = [word['y'] + word['h']//2 for word in words]
    y_tolerance = np.median([word['h'] for word in words]) * 0.5
    
    # Cluster Y positions to find rows
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
    
    # Group words by approximate X positions (columns)
    x_positions = [word['x'] + word['w']//2 for word in words]
    x_tolerance = np.median([word['w'] for word in words]) * 0.3
    
    # Cluster X positions to find columns
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
    
    # Calculate alignment score
    is_table = len(rows) >= 2 and len(columns) >= 2
    
    if is_table:
        # Check alignment quality
        row_alignment_score = 0
        col_alignment_score = 0
        
        # Check how well words align to detected rows and columns
        for word in words:
            word_y = word['y'] + word['h']//2
            word_x = word['x'] + word['w']//2
            
            # Find closest row and column
            closest_row = min(rows, key=lambda r: abs(r - word_y))
            closest_col = min(columns, key=lambda c: abs(c - word_x))
            
            # Check alignment quality
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
    
    # Method 1: Analyze alignment of table rows (if table detected)
    if table_info['is_table'] and table_info['alignment_score'] > 0.5:
        # Group words by rows
        rows = table_info['rows']
        row_tolerance = np.median([word['h'] for word in words]) * 0.5
        
        for row_y in rows:
            row_words = []
            for word in words:
                word_y = word['y'] + word['h']//2
                if abs(word_y - row_y) <= row_tolerance:
                    row_words.append(word)
            
            if len(row_words) >= 2:
                # Sort by x position
                row_words.sort(key=lambda w: w['x'])
                
                # Calculate angle from first to last word in row
                first_word = row_words[0]
                last_word = row_words[-1]
                
                y1 = first_word['y'] + first_word['h']//2
                x1 = first_word['x'] + first_word['w']//2
                y2 = last_word['y'] + last_word['h']//2
                x2 = last_word['x'] + last_word['w']//2
                
                if abs(x2 - x1) > 10:  # Avoid division by very small numbers
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    
                    # Normalize angle
                    if angle > 45:
                        angle -= 90
                    elif angle < -45:
                        angle += 90
                    
                    if abs(angle) <= 15:  # Only reasonable skew angles
                        angles.append(angle)
    
    # Method 2: Analyze text baseline using word positions
    word_centers = [(word['x'] + word['w']//2, word['y'] + word['h']//2) for word in words]
    
    if len(word_centers) >= 3:
        # Use RANSAC-like approach to find dominant text direction
        angles_sample = []
        
        for i in range(0, len(word_centers)-1, 2):
            for j in range(i+1, min(i+5, len(word_centers))):
                x1, y1 = word_centers[i]
                x2, y2 = word_centers[j]
                
                if abs(x2 - x1) > 20:  # Minimum distance
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    
                    # Normalize angle
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
    
    # Extract and store words if page number is provided
    if page_num is not None:
        extract_and_store_words(image, page_num)
    
    if not ocr_data['words']:
        return 0, 0, {'rows': [], 'columns': [], 'is_table': False, 'alignment_score': 0}
    
    # Detect table structure
    table_info = detect_table_structure(ocr_data['words'])
    
    # Base score from word count and confidence
    base_score = ocr_data['score']
    
    # Bonus for table structure detection
    if table_info['is_table']:
        structure_bonus = table_info['alignment_score'] * 50
        base_score += structure_bonus
    
    # Bonus for high confidence words
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
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None and len(lines) > 0:
            angles = []
            # Fix the unpacking issue by properly handling the lines array structure
            for line in lines[:20]:  # Consider top 20 lines
                if len(line) >= 2:
                    rho, theta = line[0]  # PaddleOCR returns lines in [[rho, theta]] format
                else:
                    rho, theta = line  # Handle different format
                
                angle = math.degrees(theta) - 90
                if abs(angle) <= 45:  # Only reasonable skew angles
                    angles.append(angle)
            
            if angles:
                # Use median angle to avoid outliers
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
            # Get OCR words for angle detection
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
        
        # Test different angles
        test_angles = np.arange(-10, 11, 0.5)
        best_angle = 0
        best_variance = 0
        
        for angle in test_angles:
            # Rotate image
            h, w = gray.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            # Calculate horizontal projection
            projection = np.sum(rotated, axis=1)
            variance = np.var(projection)
            
            if variance > best_variance:
                best_variance = variance
                best_angle = angle
        
        if abs(best_angle) <= 15:  # Only reasonable angles
            skew_angles.append(('projection_profile', best_angle))
            print(f"  Projection Profile: {best_angle:.2f}°")
    
    except Exception as e:
        print(f"  Projection Profile failed: {e}")
    
    # Method 4: Contour-based detection
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find contours
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        large_contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        if large_contours:
            angles = []
            for contour in large_contours[:10]:  # Top 10 contours
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                
                # Normalize angle
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
        # Weight different methods
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
    
    # Get image dimensions
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Create rotation matrix
    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    
    # Calculate new image dimensions after rotation
    cos_val = abs(M[0, 0])
    sin_val = abs(M[0, 1])
    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))
    
    # Adjust rotation matrix to center the image
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Perform rotation with white background
    corrected = cv2.warpAffine(image, M, (new_w, new_h), 
                              flags=cv2.INTER_CUBIC, 
                              borderMode=cv2.BORDER_CONSTANT, 
                              borderValue=(255, 255, 255))
    
    return corrected

def process_all_pages():
    """
    Main processing function that handles all pages
    """
    print("=" * 60)
    print("MULTI-OCR PDF PROCESSING WITH SKEW CORRECTION")
    print("=" * 60)
    
    # Initialize OCR engines
    initialize_ocr_engines()
    
    # Check if PDF file exists
    if not os.path.exists(pdf_file):
        print(f"Error: PDF file not found at {pdf_file}")
        return
    
    try:
        # Extract images from PDF
        page_images = extract_images_from_pdf(pdf_file)
        
        if not page_images:
            print("No pages found in PDF")
            return
        
        print(f"\nProcessing {len(page_images)} pages...")
        
        # Process each page
        corrected_images = []
        
        for page_num, original_image in page_images:
            print(f"\n--- Processing Page {page_num + 1} ---")
            
            # Detect skew angle
            skew_angle = detect_skew_comprehensive(original_image, page_num)
            
            # Correct skew
            corrected_image = correct_skew_and_enhance(original_image, skew_angle)
            
            # Final OCR extraction and storage
            print(f"  Final OCR extraction...")
            extract_and_store_words(corrected_image, page_num)
            
            corrected_images.append(corrected_image)
        
        # Save corrected images to new PDF
        save_images_to_pdf(corrected_images)
        
        # Print comprehensive summary
        print_comprehensive_summary()
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        import traceback
        traceback.print_exc()

def save_images_to_pdf(images):
    """
    Save corrected images to a new PDF file
    """
    print(f"\nSaving corrected PDF to: {output_file}")
    
    try:
        # Convert images to PIL format
        pil_images = []
        for img in images:
            # Convert BGR to RGB
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            
            pil_img = Image.fromarray(img_rgb)
            pil_images.append(pil_img)
        
        # Save as PDF
        if pil_images:
            pil_images[0].save(output_file, save_all=True, append_images=pil_images[1:], format='PDF')
            print(f"✓ Successfully saved corrected PDF with {len(pil_images)} pages")
        else:
            print("✗ No images to save")
    
    except Exception as e:
        print(f"Error saving PDF: {e}")

def print_comprehensive_summary():
    """
    Print comprehensive summary of OCR results and patterns found
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE OCR ANALYSIS SUMMARY")
    print("=" * 80)
    
    total_words = 0
    total_pages = len(extracted_words_by_page)
    
    # Page-wise summary
    print(f"\n📊 PAGE-WISE ANALYSIS ({total_pages} pages processed):")
    print("-" * 50)
    
    for page_num in sorted(extracted_words_by_page.keys()):
        page_data = extracted_words_by_page[page_num]
        words_count = page_data['word_count']
        confidence = page_data['avg_confidence']
        engine = page_data['ocr_engine']
        total_words += words_count
        
        print(f"Page {page_num}: {words_count} words | {confidence:.1f}% confidence | Engine: {engine}")
        
        # Show special patterns for this page
        if page_num in special_patterns_by_page and special_patterns_by_page[page_num]:
            patterns = special_patterns_by_page[page_num]
            pattern_summary = []
            for pattern_type, matches in patterns.items():
                pattern_summary.append(f"{pattern_type}({len(matches)})")
            print(f"         Patterns: {', '.join(pattern_summary)}")
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"   Total words extracted: {total_words}")
    print(f"   Average words per page: {total_words / total_pages:.1f}")
    
    # OCR Engine Performance
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
    
    # Special Patterns Summary
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
            unique_matches = list(dict.fromkeys(matches))  # Remove duplicates
            print(f"   {pattern_type.replace('_', ' ').title()}: {len(unique_matches)} unique")
            
            # Show first few examples
            if unique_matches:
                examples = unique_matches[:3]
                print(f"      Examples: {', '.join(str(ex) for ex in examples)}")
    else:
        print("   No special patterns detected")
    
    # OCR Performance Details
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
    print(f"   Corrected PDF saved to: {output_file}")
    
    # Save detailed results to JSON
    save_detailed_results()
    
    print("=" * 80)

def save_detailed_results():
    """
    Save detailed OCR results to JSON file for further analysis
    """
    try:
        results_data = {
            'processing_summary': {
                'total_pages': len(extracted_words_by_page),
                'total_words': sum(page['word_count'] for page in extracted_words_by_page.values()),
                'processing_date': str(np.datetime64('now')),
                'input_file': pdf_file,
                'output_file': output_file
            },
            'page_wise_results': extracted_words_by_page,
            'special_patterns': special_patterns_by_page,
            'ocr_performance': ocr_performance_by_page
        }
        
        # Save to JSON file
        json_output_file = output_file.replace('.pdf', '_analysis.json')
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"   Detailed analysis saved to: {json_output_file}")
        
    except Exception as e:
        print(f"   Warning: Could not save detailed results: {e}")

def search_extracted_text(search_term, case_sensitive=False):
    """
    Search for specific terms in extracted text across all pages
    """
    print(f"\n🔍 SEARCHING FOR: '{search_term}'")
    print("-" * 40)
    
    results = []
    
    for page_num, page_data in extracted_words_by_page.items():
        full_text = page_data['full_text']
        words_list = page_data['words_list']
        
        if not case_sensitive:
            search_text = full_text.lower()
            search_term_lower = search_term.lower()
            search_words = [w.lower() for w in words_list]
        else:
            search_text = full_text
            search_term_lower = search_term
            search_words = words_list
        
        # Search in full text
        if search_term_lower in search_text:
            # Count occurrences
            occurrences = search_text.count(search_term_lower)
            
            # Find context (surrounding words)
            words_with_term = [w for w in search_words if search_term_lower in w]
            
            results.append({
                'page': page_num,
                'occurrences': occurrences,
                'matching_words': words_with_term,
                'engine_used': page_data['ocr_engine']
            })
            
            print(f"Page {page_num}: {occurrences} occurrences | Engine: {page_data['ocr_engine']}")
            if words_with_term:
                print(f"   Matching words: {', '.join(set(words_with_term))}")
    
    if not results:
        print("No matches found")
    else:
        total_occurrences = sum(r['occurrences'] for r in results)
        print(f"\nTotal: {total_occurrences} occurrences across {len(results)} pages")
    
    return results

def export_extracted_text(format_type='txt'):
    """
    Export all extracted text to different formats
    """
    if not extracted_words_by_page:
        print("No extracted text to export")
        return
    
    base_name = output_file.replace('.pdf', '')
    
    if format_type.lower() == 'txt':
        # Export as plain text
        txt_file = f"{base_name}_extracted_text.txt"
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            for page_num in sorted(extracted_words_by_page.keys()):
                page_data = extracted_words_by_page[page_num]
                f.write(f"=== PAGE {page_num} ===\n")
                f.write(f"OCR Engine: {page_data['ocr_engine']}\n")
                f.write(f"Word Count: {page_data['word_count']}\n")
                f.write(f"Confidence: {page_data['avg_confidence']:.1f}%\n\n")
                f.write(page_data['full_text'])
                f.write("\n\n" + "="*50 + "\n\n")
        
        print(f"✓ Text exported to: {txt_file}")
    
    elif format_type.lower() == 'csv':
        # Export as CSV with word details
        import csv
        csv_file = f"{base_name}_word_details.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Page', 'Word', 'Confidence', 'X', 'Y', 'Width', 'Height', 'OCR_Engine'])
            
            for page_num in sorted(extracted_words_by_page.keys()):
                page_data = extracted_words_by_page[page_num]
                
                for word_detail in page_data.get('word_details', []):
                    writer.writerow([
                        page_num,
                        word_detail['text'],
                        word_detail['confidence'],
                        word_detail['x'],
                        word_detail['y'],
                        word_detail['width'],
                        word_detail['height'],
                        word_detail['engine']
                    ])
        
        print(f"✓ Word details exported to: {csv_file}")

# Additional utility functions for advanced analysis
def analyze_document_structure():
    """
    Analyze overall document structure and provide insights
    """
    print(f"\n📋 DOCUMENT STRUCTURE ANALYSIS:")
    print("-" * 40)
    
    if not extracted_words_by_page:
        print("No data available for analysis")
        return
    
    # Analyze word count distribution
    word_counts = [page['word_count'] for page in extracted_words_by_page.values()]
    
    print(f"Word count statistics:")
    print(f"   Min: {min(word_counts)} words")
    print(f"   Max: {max(word_counts)} words")
    print(f"   Mean: {np.mean(word_counts):.1f} words")
    print(f"   Median: {np.median(word_counts):.1f} words")
    print(f"   Std Dev: {np.std(word_counts):.1f}")
    
    # Identify potential header/footer pages
    avg_words = np.mean(word_counts)
    threshold = avg_words * 0.3  # Pages with less than 30% of average might be headers/footers
    
    sparse_pages = [page_num for page_num, page_data in extracted_words_by_page.items() 
                   if page_data['word_count'] < threshold]
    
    if sparse_pages:
        print(f"   Potential header/footer pages: {sparse_pages}")
    
    # Analyze confidence trends
    confidences = [page['avg_confidence'] for page in extracted_words_by_page.values()]
    
    print(f"\nConfidence statistics:")
    print(f"   Average confidence: {np.mean(confidences):.1f}%")
    print(f"   Confidence range: {min(confidences):.1f}% - {max(confidences):.1f}%")
    
    # Find pages with low confidence
    low_conf_threshold = 70
    low_conf_pages = [page_num for page_num, page_data in extracted_words_by_page.items() 
                     if page_data['avg_confidence'] < low_conf_threshold]
    
    if low_conf_pages:
        print(f"   Pages with low confidence (<{low_conf_threshold}%): {low_conf_pages}")

# Main execution
if __name__ == "__main__":
    print("Starting Multi-OCR PDF Processing...")
    print(f"Input PDF: {pdf_file}")
    print(f"Output PDF: {output_file}")
    
    # Process all pages
    process_all_pages()
    
    # Additional analysis
    analyze_document_structure()
    
    # Export options
    print(f"\n📤 EXPORT OPTIONS:")
    print("   Run export_extracted_text('txt') for plain text export")
    print("   Run export_extracted_text('csv') for detailed CSV export")
    print("   Run search_extracted_text('your_term') to search extracted text")
    
    print(f"\n✅ Processing completed successfully!")
    print(f"   Check output file: {output_file}")
