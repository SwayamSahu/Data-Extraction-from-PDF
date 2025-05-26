import fitz
import pytesseract
from PIL import Image
import io
import math

# Standard page sizes (in points, at 72 DPI)
STANDARD_SIZES = {
    'A4': (595, 842),
    'Letter': (612, 792),
    'Legal': (612, 1008),
    'A3': (842, 1191)
}

def get_rotation_angle(image):
    try:
        osd = pytesseract.image_to_osd(image)
        angle = 0
        for line in osd.splitlines():
            if "Rotate" in line:
                angle = int(line.split(":")[-1].strip())
                break
        return angle
    except:
        return 0  # Default to no rotation if OCR fails

def rotate_image(image, angle):
    if angle == 0:
        return image
    return image.rotate(-angle, expand=True)  # expand to avoid cropping

def scale_and_center_image(image, target_width, target_height):
    """Scale image to fit within target dimensions while preserving aspect ratio, then center it"""
    original_width, original_height = image.size
    scale = min(target_width/original_width, target_height/original_height)
    
    if scale < 1:  # Only scale down if needed
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    else:
        new_width, new_height = original_width, original_height
    
    # Create new image with target dimensions
    background = Image.new("RGB", (target_width, target_height), (255, 255, 255))
    x = (target_width - new_width) // 2
    y = (target_height - new_height) // 2
    background.paste(image, (x, y))
    
    return background

def correct_pdf_alignment(input_pdf_path, output_pdf_path, page_size='A4', dpi=300):
    """
    Correct alignment of scanned PDF pages while preserving all content.
    
    Args:
        input_pdf_path: Path to input PDF file
        output_pdf_path: Path to save corrected PDF
        page_size: One of 'A4', 'Letter', 'Legal', 'A3' (default: 'A4')
        dpi: Resolution for image processing (default: 300)
    """
    if page_size not in STANDARD_SIZES:
        raise ValueError(f"Invalid page size. Choose from: {list(STANDARD_SIZES.keys())}")
    
    # Convert standard size from points to pixels at specified DPI
    points_to_pixels = dpi / 72
    target_width = int(STANDARD_SIZES[page_size][0] * points_to_pixels)
    target_height = int(STANDARD_SIZES[page_size][1] * points_to_pixels)
    
    pdf = fitz.open(input_pdf_path)
    processed_images = []

    for page_num in range(len(pdf)):
        page = pdf.load_page(page_num)
        
        # Get page content as high-resolution image
        zoom = dpi / 72  # fitz uses 72 DPI as base
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # Correct orientation
        angle = get_rotation_angle(img)
        rotated_img = rotate_image(img, angle)
        
        # Scale to fit standard size while preserving content
        scaled_img = scale_and_center_image(rotated_img, target_width, target_height)
        processed_images.append(scaled_img)

    # Save all pages to PDF
    if processed_images:
        processed_images[0].save(
            output_pdf_path,
            save_all=True,
            append_images=processed_images[1:],
            dpi=(dpi, dpi),
            quality=100
        )

    print(f"Corrected PDF saved to: {output_pdf_path}")

# Usage example:
correct_pdf_alignment("aligned_output.pdf", "A3.pdf", page_size='A4')

