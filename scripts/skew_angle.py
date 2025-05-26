# PDF : Skew Angle,Rotation and OCR

import fitz  # PyMuPDF
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pytesseract
from PIL import Image
import os

# Step 1: Set up Tesseract command path
pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\SwayamPrakashSahu\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"

# Step 2: Update file path (Replace with the actual file path in Google Drive)
#sarfaraz
#pdf_file = '/content/drive/MyDrive/project/image_processing/505150_3030065986_1060_BOL_removed.pdf'  # Input PDF
#output_file = '/content/drive/MyDrive/project/image_processing/505150_3030065986_1060_BOL_corrected.pdf'  # Output PDF

#sashi
pdf_file = r'C:\Users\SwayamPrakashSahu\Downloads\AMAT_TableExtractor\input_pdfs\505150_3030065986_1060_BOL (2).pdf'  # Input PDF
output_file = r'C:\Users\SwayamPrakashSahu\Downloads\AMAT_TableExtractor\corrected_output.pdf'  # Output PDF

# Function to extract images from a PDF
def extract_images_from_pdf(pdf_path):
    print("Inside extract_images_from_pdf")
    """
    Render each page of a PDF as a single image.
    """
    page_images = []  # List to store rendered page images
    pdf = fitz.open(pdf_path)  # Open the PDF document

    for page_num in range(len(pdf)):
        page = pdf[page_num]

        # Render the entire page to a pixmap at a high resolution
        pix = page.get_pixmap(dpi=300)  # Increase DPI for better quality
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        #plt.figure(figsize=(6,6))
        #plt.imshow(image, cmap='gray')
        #plt.title('pix')
        #plt.axis('off')
        #plt.show()

        # Convert grayscale to BGR for uniformity
        if pix.n < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
         # Optional: Display the rendered page image
        page_images.append((page_num, image))
        print("Outside extract_images_from_pdf")

    pdf.close()
    return page_images

# Function to calculate skew angle of an image

def calculate_skew_angle(image, rotation_count=0, max_rotations=4):
    print("Inside calculate_skew_angle")

    # Step 1: Convert to grayscalex
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Adaptive Thresholding for better text isolation
    #1 binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 3: Apply Morphological Operations for better text region detection
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, noise_kernel, iterations=1)

    # Step 4: Dynamically Adjust Kernel Size Based on Image Size
    height, width = image.shape[:2]

    # Adaptive kernel size based on image dimensions
    kernel_width = max(2, width // 200)
    kernel_height = max(5, height // 150)

    # Step 5: Use a Structuring Element Optimized for Text Detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))

    # Step 6: Apply Closing to Connect Broken Characters and Improve OCR
    dilated = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Check the image
    plt.figure(figsize=(10 ,8))
    plt.imshow(dilated, cmap='gray')
    plt.title('dilated')
    plt.axis('off')
    plt.show()


    # Step 4: Run OCR on the processed image
    #psm 3	Default, good for mixed layouts
    #psm 6	Best for paragraph text (Standard text in PDFs)
    #psm 4	Works better for tilted/skewed text
    #psm 11	Single-line text only
    #psm 5	Works best for vertical text

    custom_config = r'--oem 3 --psm 5'  # PSM 5: Treats text as a single vertical block
    data = pytesseract.image_to_data(dilated, config=custom_config, output_type=pytesseract.Output.DICT)

    #Save the Words extracted in a String Array. Some regular expression to identify atleast 2 words.Include PaddleOCR and EasyOCR and also check confidence of words.

    # Print OCR data for debugging
    print("OCR Data:", data)

    # Extract bounding boxes for detected text
    text_boxes = []
    detected_words = []

    for i in range(len(data['text'])):
        if data['text'][i].strip():  # Ensure text is detected
            
            print(f"Word: '{data['text'][i]}'- Confidence: {data['conf'][i]}")
            conf = int(data['conf'][i])

            # Keep only high-confidence words
            if conf > 50:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                text_boxes.append((x, y, w, h))
                detected_words.append(data['text'][i])

    # Step 5: If less than 5 words are detected, rotate the image and retry OCR
    if len(detected_words) < 5:
       if rotation_count < max_rotations: 
           print("Few words detected. Rotating image and retrying OCR...")
           rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
           return calculate_skew_angle(rotated, rotation_count + 1, max_rotations)  # Recursive call with rotated image
       else:
           print("Maximum rotations reached. Skipping further rotation.")
           return 0

    # Step 6: Draw bounding boxes on detected words for visualization
    mask = np.zeros_like(gray)
    for (x, y, w, h) in text_boxes:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # Step 7: Detect edges and use Hough Transform to find skew
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is None:
        return 0  # No lines detected, assume no skew

    # Step 8: Compute skew angles from detected lines
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90  # Convert to degree format

        # Ignore near-horizontal and near-vertical lines
        if -10 < angle < 10 or 80 < abs(angle) < 100:
            continue

        angles.append(angle)

    # Step 9: Compute the median skew angle
    if not angles:
        return 0
    median_angle = np.median(angles)

    # Step 10: If the detected skew is very high (near ±90°), assume text is vertical/right-aligned, stop rotating
    if abs(median_angle) > 85:
        print("Text is right-aligned. No further rotation needed.")
        return 0

    print(f"Detected Skew Angle: {median_angle:.2f}°")
    print("Outside calculate_skew_angle")
    return median_angle

# Function to rotate an image by a given angle
def rotate_image(image, angle):
    print("Inside rotate_image")
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Calculate the bounding box of the rotated image
    abs_cos = abs(math.cos(math.radians(angle)))
    abs_sin = abs(math.sin(math.radians(angle)))

    # New width and height bounds
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # Adjust the rotation matrix to take into account the new dimensions
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation with the adjusted bounds
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    print("Outside rotate_image")
    return rotated

# Function to process a PDF for skew detection and correction
def correct_pdf_skew(pdf_path, output_path):
    print("inside correct_pdf_skew")
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    results = []
    images = extract_images_from_pdf(pdf_path)
    corrected_images = []

    for page_num, image in images:
        angle = calculate_skew_angle(image)
        print('page number =',page_num,'angle = ',angle)
        rotated_image = rotate_image(image, angle)  # Rotate by the positive angle
        rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)# Convert corrected image to PIL format for saving as PDF
        pil_image = Image.fromarray(rotated_image)
        corrected_images.append(pil_image)
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Page {page_num + 1}")
        plt.axis('off')
        plt.show()
        results.append((page_num, angle))

    # Save corrected images as a new PDF
    corrected_images[0].save(
        output_path, save_all=True, append_images=corrected_images[1:]
    )
    print(f"Corrected PDF saved to: {output_path}")
    print("Outside correct_pdf_skew")
    return results

# Main Execution
if __name__ == "__main__":
    print("Inside Main")
    tilt_results = correct_pdf_skew(pdf_file, output_file)
    for page_num, tilt_angle in tilt_results:
        print(f"Page {page_num + 1}: Tilt Angle = {tilt_angle:.2f} degrees")
    print("Outside Main")
