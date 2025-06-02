# PDF : Skew Angle,Rotation and OCR - Complete Fix - Enhanced for All Table Types
# With Word Extraction and Pattern Recognition

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

# Step 1: Set up Tesseract command path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Step 2: Update file path
pdf_file = r'C:\Users\hp\Downloads\amat_final_work\Data-Extraction-from-PDF\input_pdfs\505150_3030065986_1060_BOL (2).pdf'
output_file = r'C:\Users\hp\Downloads\amat_final_work\Data-Extraction-from-PDF\corrected3_output.pdf'

# Global storage for extracted words
extracted_words_by_page = {}
special_patterns_by_page = {}

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
    Extract words from image and store them page-wise
    Also identify special patterns
    """
    global extracted_words_by_page, special_patterns_by_page
    
    try:
        processed = preprocess_for_ocr(image)
        
        # Get text using different methods
        # Method 1: Full text extraction
        full_text = pytesseract.image_to_string(processed, config=r'--oem 3 --psm 3')
        
        # Method 2: Word-level extraction with positions
        word_data = pytesseract.image_to_data(processed, config=r'--oem 3 --psm 3', output_type=pytesseract.Output.DICT)
        
        # Extract individual words
        words_list = []
        word_details = []
        
        for i in range(len(word_data['text'])):
            text = word_data['text'][i].strip()
            conf = int(word_data['conf'][i])
            
            # Filter valid words
            if (text and len(text) > 0 and conf > 30 and 
                re.match(r'^[a-zA-Z0-9\s\-\.\,\:\(\)\[\]\/\&\%\$\#\@\!\+\=\*]+', text)):
                
                words_list.append(text)
                word_details.append({
                    'text': text,
                    'confidence': conf,
                    'x': word_data['left'][i],
                    'y': word_data['top'][i],
                    'width': word_data['width'][i],
                    'height': word_data['height'][i],
                    'page': page_num + 1
                })
        
        # Store words for this page
        extracted_words_by_page[page_num + 1] = {
            'full_text': full_text,
            'words_list': words_list,
            'word_details': word_details,
            'word_count': len(words_list)
        }
        
        # Identify special patterns
        special_patterns = identify_special_patterns(words_list)
        special_patterns_by_page[page_num + 1] = special_patterns
        
        print(f"Page {page_num + 1}: Extracted {len(words_list)} words")
        if special_patterns:
            print(f"  Found special patterns: {list(special_patterns.keys())}")
        
        return words_list, word_details, special_patterns
        
    except Exception as e:
        print(f"Word extraction error for page {page_num + 1}: {e}")
        extracted_words_by_page[page_num + 1] = {
            'full_text': '',
            'words_list': [],
            'word_details': [],
            'word_count': 0
        }
        special_patterns_by_page[page_num + 1] = {}
        return [], [], {}

def get_detailed_ocr_data(image):
    """Get detailed OCR data including bounding boxes and confidence"""
    try:
        processed = preprocess_for_ocr(image)
        
        # Try multiple OCR configurations
        configs = [
            r'--oem 3 --psm 5',
            r'--oem 3 --psm 3',  # Default
            r'--oem 3 --psm 6',  # Uniform block
            r'--oem 3 --psm 4',  # Single column
            r'--oem 3 --psm 1'   # Automatic with OSD
        ]
        
        best_data = None
        best_score = 0
        
        for config in configs:
            try:
                data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
                
                # Calculate score based on valid words and confidence
                valid_words = []
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    conf = int(data['conf'][i])
                    
                    if (text and len(text) > 1 and conf > 30 and 
                        re.match(r'^[a-zA-Z0-9\s\-\.\,\:\(\)\[\]\/\&\%\$\#\@\!]+', text)):
                        valid_words.append({
                            'text': text,
                            'confidence': conf,
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'w': data['width'][i],
                            'h': data['height'][i]
                        })
                
                if valid_words:
                    avg_conf = np.mean([w['confidence'] for w in valid_words])
                    score = len(valid_words) * (avg_conf / 100)
                    
                    if score > best_score:
                        best_score = score
                        best_data = {
                            'words': valid_words,
                            'score': score,
                            'word_count': len(valid_words),
                            'avg_confidence': avg_conf
                        }
                        
            except Exception as e:
                continue
        
        return best_data if best_data else {'words': [], 'score': 0, 'word_count': 0, 'avg_confidence': 0}
        
    except Exception as e:
        print(f"OCR Error: {e}")
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
        return 0, 0, 0, None, None
    
    # Detect table structure
    table_info = detect_table_structure(ocr_data['words'])
    
    # Detect text angles
    text_angles = detect_text_angles_from_structure(ocr_data['words'], table_info)
    
    # Enhanced scoring
    base_score = ocr_data['score']
    
    # Bonus for table structure detection
    if table_info['is_table']:
        structure_bonus = table_info['alignment_score'] * 0.5 * len(ocr_data['words'])
        base_score += structure_bonus
    
    # Penalty for inconsistent text angles (indicates wrong orientation)
    if text_angles:
        angle_std = np.std(text_angles)
        if angle_std > 5:  # High variation in angles suggests wrong orientation
            base_score *= 0.7
    
    return (base_score, ocr_data['word_count'], ocr_data['avg_confidence'], 
            table_info, text_angles)

def find_best_orientation(image, page_num=None):
    """Find the best orientation by testing all 4 rotations with enhanced scoring"""
    print("Testing orientations with enhanced table detection...")
    
    orientations = {
        0: image,
        90: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(image, cv2.ROTATE_180),
        270: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    }
    
    best_rotation = 0
    best_score = 0
    scores = {}
    
    for rotation, rotated_img in orientations.items():
        # Only extract words for the best orientation (determined later)
        temp_page_num = page_num if rotation == 0 else None
        score, word_count, confidence, table_info, text_angles = enhanced_get_ocr_score(rotated_img, temp_page_num)
        
        scores[rotation] = {
            'score': score,
            'words': word_count,
            'confidence': confidence,
            'table_detected': table_info['is_table'] if table_info else False,
            'table_alignment': table_info['alignment_score'] if table_info else 0,
            'table_structure': f"{table_info['row_count']}x{table_info['col_count']}" if table_info else "0x0",
            'text_angles': text_angles if text_angles else []
        }
        
        print(f"Rotation {rotation}°:")
        print(f"  Score={score:.2f}, Words={word_count}, Conf={confidence:.1f}")
        if table_info:
            print(f"  Table: {table_info['is_table']} ({table_info['row_count']}x{table_info['col_count']}, alignment={table_info['alignment_score']:.2f})")
        if text_angles:
            print(f"  Text angles: {[f'{a:.1f}°' for a in text_angles[:5]]}")
        
        if score > best_score:
            best_score = score
            best_rotation = rotation
    
    # Extract words from the best orientation
    if page_num is not None and best_rotation != 0:
        best_image = orientations[best_rotation]
        extract_and_store_words(best_image, page_num)
    
    print(f"Best orientation: {best_rotation}° (Score: {best_score:.2f})")
    return best_rotation, scores

def detect_fine_skew(image):
    """Detect fine skew angle using both line detection and text structure"""
    print("Detecting fine skew with enhanced methods...")
    
    # Get OCR data for structure analysis
    ocr_data = get_detailed_ocr_data(image)
    table_info = detect_table_structure(ocr_data['words'])
    text_angles = detect_text_angles_from_structure(ocr_data['words'], table_info)
    
    angles = []
    
    # Method 1: Text structure analysis (primary for tables without lines)
    if text_angles:
        # Use median of detected text angles
        median_text_angle = np.median(text_angles)
        angles.append(median_text_angle)
        print(f"Text structure angle: {median_text_angle:.2f}°")
    
    # Method 2: Traditional line detection (for tables with lines)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = preprocess_for_ocr(gray)
    
    # Detect horizontal lines
    kernel_length = max(1, image.shape[1] // 50)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) >= 2:
        min_area = (image.shape[0] * image.shape[1]) * 0.0001
        line_angles = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                angle = math.degrees(math.atan2(vy, vx))
                
                if angle > 45:
                    angle -= 90
                elif angle < -45:
                    angle += 90
                
                if abs(angle) <= 15:
                    line_angles.append(angle)
        
        if line_angles:
            median_line_angle = np.median(line_angles)
            angles.append(median_line_angle)
            print(f"Line detection angle: {median_line_angle:.2f}°")
    
    # Method 3: Hough line detection for additional validation
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=int(image.shape[1]*0.3))
    
    if lines is not None:
        hough_angles = []
        for line in lines[:10]:  # Use top 10 lines
            rho, theta = line[0]
            angle = math.degrees(theta - np.pi/2)
            
            if angle > 45:
                angle -= 90
            elif angle < -45:
                angle += 90
            
            if abs(angle) <= 15:
                hough_angles.append(angle)
        
        if hough_angles:
            median_hough_angle = np.median(hough_angles)
            angles.append(median_hough_angle)
            print(f"Hough line angle: {median_hough_angle:.2f}°")
    
    # Combine all angle measurements
    if not angles:
        print("No skew detected from any method")
        return 0
    
    # Use weighted average, giving more weight to text structure for space-separated tables
    if table_info and table_info['is_table'] and table_info['alignment_score'] > 0.6:
        # For well-structured tables, prioritize text structure
        if text_angles:
            final_angle = np.median(text_angles)
        else:
            final_angle = np.median(angles)
    else:
        # For other content, use all methods
        final_angle = np.median(angles)
    
    print(f"Final skew angle: {final_angle:.2f}°")
    return final_angle

def rotate_image(image, angle):
    """Rotate image by given angle with proper bounds calculation"""
    if abs(angle) < 0.1:
        return image
    
    print(f"Applying rotation: {angle:.2f}°")
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Calculate new dimensions to prevent cropping
    angle_rad = math.radians(abs(angle))
    new_w = int(w * math.cos(angle_rad) + h * math.sin(angle_rad))
    new_h = int(w * math.sin(angle_rad) + h * math.cos(angle_rad))
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Adjust translation to center the image
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    # Apply rotation with white background
    rotated = cv2.warpAffine(image, M, (new_w, new_h),
                           flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255, 255, 255))
    
    return rotated

def standardize_dimensions(images, target_width=None, target_height=None):
    """Standardize all images to same dimensions"""
    print("Standardizing dimensions...")
    
    if not images:
        return images
    
    # Find maximum dimensions if not specified
    if target_width is None or target_height is None:
        max_w = max(img.shape[1] for img in images)
        max_h = max(img.shape[0] for img in images)
        target_width = target_width or max_w
        target_height = target_height or max_h
    
    standardized = []
    for img in images:
        h, w = img.shape[:2]
        
        # Create white canvas of target size
        canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
        
        # Calculate position to center the image
        y_offset = (target_height - h) // 2
        x_offset = (target_width - w) // 2
        
        # Ensure offsets are non-negative
        y_offset = max(0, y_offset)
        x_offset = max(0, x_offset)
        
        # Calculate actual dimensions to place
        end_y = min(y_offset + h, target_height)
        end_x = min(x_offset + w, target_width)
        img_h = end_y - y_offset
        img_w = end_x - x_offset
        
        # Place image on canvas
        canvas[y_offset:end_y, x_offset:end_x] = img[:img_h, :img_w]
        standardized.append(canvas)
    
    print(f"Standardized to {target_width}x{target_height}")
    return standardized

def correct_page_orientation(image, page_num):
    """Complete orientation correction for a single page"""
    print(f"Correcting page orientation...")
    
    # Step 1: Find best major rotation using enhanced scoring
    best_rotation, scores = find_best_orientation(image, page_num)
    
    # Step 2: Apply major rotation
    if best_rotation == 90:
        corrected = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif best_rotation == 180:
        corrected = cv2.rotate(image, cv2.ROTATE_180)
    elif best_rotation == 270:
        corrected = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        corrected = image.copy()
    
    # Step 3: Detect and correct fine skew using enhanced methods
    fine_skew = detect_fine_skew(corrected)
    
    # Step 4: Apply fine rotation if significant
    if abs(fine_skew) > 0.5:
        corrected = rotate_image(corrected, -fine_skew)  # Negative to correct
        # Re-extract words after fine correction
        extract_and_store_words(corrected, page_num)
    
    return corrected, best_rotation, fine_skew, scores

def print_extracted_words():
    """Print all extracted words page-wise"""
    global extracted_words_by_page, special_patterns_by_page
    
    print(f"\n{'='*80}")
    print("EXTRACTED WORDS AND PATTERNS SUMMARY")
    print(f"{'='*80}")
    
    total_words = 0
    total_patterns = 0
    
    for page_num in sorted(extracted_words_by_page.keys()):
        page_data = extracted_words_by_page[page_num]
        patterns = special_patterns_by_page.get(page_num, {})
        
        print(f"\n{'='*60}")
        print(f"PAGE {page_num}")
        print(f"{'='*60}")
        
        print(f"Total Words Extracted: {page_data['word_count']}")
        total_words += page_data['word_count']
        
        # Print first 50 words as sample
        if page_data['words_list']:
            print(f"\nWords Array (showing first 50):")
            sample_words = page_data['words_list'][:50]
            print(f"Words: {sample_words}")
            
            if len(page_data['words_list']) > 50:
                print(f"... and {len(page_data['words_list']) - 50} more words")
        
        # Print special patterns found
        if patterns:
            print(f"\nSpecial Patterns Identified:")
            pattern_count = 0
            for pattern_type, matches in patterns.items():
                if matches:
                    print(f"\n  {pattern_type.upper().replace('_', ' ')}:")
                    for match in matches[:10]:  # Show first 10 matches
                        print(f"    - {match}")
                        pattern_count += 1
                    if len(matches) > 10:
                        print(f"    ... and {len(matches) - 10} more {pattern_type}")
            
            total_patterns += pattern_count
            print(f"\nTotal patterns found on this page: {pattern_count}")
        else:
            print("\nNo special patterns identified on this page.")
        
        # Print confidence statistics
        if page_data['word_details']:
            confidences = [word['confidence'] for word in page_data['word_details']]
            avg_confidence = np.mean(confidences)
            min_confidence = min(confidences)
            max_confidence = max(confidences)
            
            print(f"\nOCR Confidence Statistics:")
            print(f"  Average: {avg_confidence:.1f}%")
            print(f"  Range: {min_confidence}% - {max_confidence}%")
    
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Total Pages Processed: {len(extracted_words_by_page)}")
    print(f"Total Words Extracted: {total_words}")
    print(f"Total Special Patterns Found: {total_patterns}")
    
    # Summary of all unique patterns across all pages
    all_patterns = {}
    for page_patterns in special_patterns_by_page.values():
        for pattern_type, matches in page_patterns.items():
            if pattern_type not in all_patterns:
                all_patterns[pattern_type] = []
            all_patterns[pattern_type].extend(matches)
    
    # Remove duplicates
    for pattern_type in all_patterns:
        all_patterns[pattern_type] = list(set(all_patterns[pattern_type]))
    
    if all_patterns:
        print(f"\nUnique Patterns Found Across All Pages:")
        for pattern_type, unique_matches in all_patterns.items():
            if unique_matches:
                print(f"  {pattern_type.upper().replace('_', ' ')}: {len(unique_matches)} unique items")

def save_extracted_data_to_files():
    """Save extracted words and patterns to separate files"""
    global extracted_words_by_page, special_patterns_by_page
    
    try:
        # Save words data
        words_file = output_file.replace('.pdf', '_extracted_words.json')
        with open(words_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_words_by_page, f, indent=2, ensure_ascii=False)
        print(f"Words data saved to: {words_file}")
        
        # Save patterns data
        patterns_file = output_file.replace('.pdf', '_special_patterns.json')
        with open(patterns_file, 'w', encoding='utf-8') as f:
            json.dump(special_patterns_by_page, f, indent=2, ensure_ascii=False)
        print(f"Patterns data saved to: {patterns_file}")
        
        # Save summary text file
        summary_file = output_file.replace('.pdf', '_extraction_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("PDF TEXT EXTRACTION SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            total_words = 0
            for page_num in sorted(extracted_words_by_page.keys()):
                page_data = extracted_words_by_page[page_num]
                patterns = special_patterns_by_page.get(page_num, {})
                
                f.write(f"PAGE {page_num}\n")
                f.write("-" * 20 + "\n")
                f.write(f"Words extracted: {page_data['word_count']}\n")
                total_words += page_data['word_count']
                
                # Write all words
                f.write(f"Words: {', '.join(page_data['words_list'])}\n\n")
                
                # Write patterns
                if patterns:
                    f.write("Special Patterns:\n")
                    for pattern_type, matches in patterns.items():
                        if matches:
                            f.write(f"  {pattern_type}: {', '.join(map(str, matches))}\n")
                    f.write("\n")
                
                f.write(f"Full Text:\n{page_data['full_text']}\n")
                f.write("\n" + "="*50 + "\n\n")
            
            f.write(f"TOTAL WORDS EXTRACTED: {total_words}\n")
        
        print(f"Summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"Error saving extracted data: {e}")

def correct_pdf_skew(pdf_path, output_path):
    """Main function to correct PDF skew and orientation"""
    print("Starting PDF correction with enhanced table detection and word extraction...")
    
    # Create output directory
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract pages
    page_images = extract_images_from_pdf(pdf_path)
    corrected_images = []
    results = []
    
    print(f"\nProcessing {len(page_images)} pages...")
    
    for page_num, original_image in page_images:
        print(f"\n{'='*50}")
        print(f"PROCESSING PAGE {page_num + 1}")
        print(f"{'='*50}")
        
        # Show original
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Original Page {page_num + 1}")
        plt.axis('off')
        plt.show()
        
        # Correct orientation and extract words
        corrected_image, major_rotation, fine_skew, orientation_scores = correct_page_orientation(original_image, page_num)
        
        # Show corrected
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Corrected Page {page_num + 1} (Major: {major_rotation}°, Fine: {fine_skew:.2f}°)")
        plt.axis('off')
        plt.show()
        
        corrected_images.append(corrected_image)
        results.append({
            'page': page_num + 1,
            'major_rotation': major_rotation,
            'fine_skew': fine_skew,
            'scores': orientation_scores
        })
    
    # Print extracted words and patterns
    print_extracted_words()
    
    # Save extracted data to files
    save_extracted_data_to_files()
    
    # Standardize dimensions
    print(f"\nStandardizing dimensions...")
    standardized_images = standardize_dimensions(corrected_images)
    
    # Convert to PIL and save
    print(f"Saving corrected PDF...")
    pil_images = []
    for img in standardized_images:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        pil_images.append(pil_img)
    
    # Save as PDF
    if pil_images:
        pil_images[0].save(
            output_path, 
            save_all=True, 
            append_images=pil_images[1:],
            format='PDF',
            resolution=300.0,
            quality=95
        )
        print(f"Corrected PDF saved to: {output_path}")
    
    return results

# Additional utility functions for word analysis
def get_words_by_pattern(pattern_type):
    """Get all words matching a specific pattern across all pages"""
    global special_patterns_by_page
    
    all_matches = []
    for page_num, patterns in special_patterns_by_page.items():
        if pattern_type in patterns:
            all_matches.extend(patterns[pattern_type])
    
    return list(set(all_matches))  # Remove duplicates

def get_words_by_page(page_num):
    """Get all words from a specific page"""
    global extracted_words_by_page
    
    if page_num in extracted_words_by_page:
        return extracted_words_by_page[page_num]['words_list']
    return []

def search_words(search_term, case_sensitive=False):
    """Search for specific words across all pages"""
    global extracted_words_by_page
    
    results = {}
    
    for page_num, page_data in extracted_words_by_page.items():
        matches = []
        for word in page_data['words_list']:
            if case_sensitive:
                if search_term in word:
                    matches.append(word)
            else:
                if search_term.lower() in word.lower():
                    matches.append(word)
        
        if matches:
            results[page_num] = matches
    
    return results

# Main execution
if __name__ == "__main__":
    print("Enhanced PDF Orientation Correction Tool with Word Extraction")
    print("Supports both grid-based and space-separated tables")
    print("Includes pattern recognition for special content")
    print("="*60)
    
    try:
        # Process PDF
        results = correct_pdf_skew(pdf_file, output_file)
        
        # Print summary
        print(f"\n{'='*60}")
        print("CORRECTION SUMMARY")
        print(f"{'='*60}")
        
        total_major_rotations = 0
        total_fine_corrections = 0
        
        for result in results:
            page = result['page']
            major = result['major_rotation']
            fine = result['fine_skew']
            scores = result['scores']
            
            print(f"\nPage {page}:")
            print(f"  Major rotation: {major}°")
            print(f"  Fine skew: {fine:.2f}°")
            print(f"  Total correction: {major + fine:.2f}°")
            
            # Show enhanced orientation analysis
            print(f"  Orientation analysis:")
            for angle, score_data in scores.items():
                table_str = f"Table: {score_data['table_structure']}" if score_data['table_detected'] else "No table"
                print(f"    {angle}°: Score={score_data['score']:.2f}, Words={score_data['words']}, {table_str}")
            
            if major != 0:
                total_major_rotations += 1
            if abs(fine) > 0.5:
                total_fine_corrections += 1
        
        print(f"\n{'='*60}")
        print(f"STATISTICS:")
        print(f"  Total pages processed: {len(results)}")
        print(f"  Pages with major rotation: {total_major_rotations}")
        print(f"  Pages with fine correction: {total_fine_corrections}")
        
        # Word extraction statistics
        total_words = sum(page_data['word_count'] for page_data in extracted_words_by_page.values())
        total_patterns = sum(len([m for matches in page_patterns.values() for m in matches]) 
                           for page_patterns in special_patterns_by_page.values())
        
        print(f"  Total words extracted: {total_words}")
        print(f"  Total special patterns found: {total_patterns}")
        print(f"  Output file: {output_file}")
        print(f"{'='*60}")
        print("Enhanced PDF correction with word extraction completed successfully!")
        
        # Example usage of utility functions
        print(f"\n{'='*60}")
        print("EXAMPLE SEARCHES:")
        print(f"{'='*60}")
        
        # Search for emails
        emails = get_words_by_pattern('emails')
        if emails:
            print(f"Found emails: {emails}")
        
        # Search for phone numbers
        phones = get_words_by_pattern('phone_numbers')
        if phones:
            print(f"Found phone numbers: {phones}")
        
        # Search for specific terms (example)
        search_results = search_words('total', case_sensitive=False)
        if search_results:
            print(f"Pages containing 'total': {list(search_results.keys())}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
