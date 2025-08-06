import os
from paddleocr import PaddleOCR
import pypdfium2 as pdfium

# === 1. Config ===
PDF_PATH = r"C:\Users\SwayamPrakashSahu\Downloads\AMAT_TableExtractor\input_pdfs\505150_3030065986_1060_BOL (2).pdf"
OUTPUT_DIR = "output"

# === 2. Setup ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

# === 3. Load and render PDF ===
pdf = pdfium.PdfDocument(PDF_PATH)
for page_num in range(len(pdf)):
    page = pdf[page_num]
    image = page.render(scale=3).to_pil()
    image_path = os.path.join(OUTPUT_DIR, f"page_{page_num}.jpg")
    image.save(image_path)

    # === 4. Run OCR ===
    result = ocr.predict(input=image_path)

    # === 5. Save results ===
    for res_id, res in enumerate(result):
        # Corrected saving functions with full paths
        img_save_path = os.path.join(OUTPUT_DIR, f"page_{page_num}_result_{res_id}.jpg")
        res.save_to_img(img_save_path)  # Provide full image path
        
        json_save_path = os.path.join(OUTPUT_DIR, f"page_{page_num}_result_{res_id}.json")
        res.save_to_json(json_save_path)  # Provide full JSON path

    print(f"[✓] Processed page {page_num + 1}/{len(pdf)}")

print("\n🎉 All pages processed. Check the 'output' folder.")
