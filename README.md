# 📄 PDF to Table Extractor

A robust, fault-tolerant tool to extract all tables from both native and scanned PDF files into CSV format.

## 🚀 Features

- ✅ **Auto-detects** if a PDF is native (contains selectable text) or scanned (image-only).
- 📑 **Native PDFs**:
  - Uses `pdfplumber` for high-fidelity table parsing.
- 📷 **Scanned PDFs**:
  - Detects tables using [Microsoft's Table Transformer model](https://huggingface.co/microsoft/table-transformer-detection).
  - OCRs the tables using `Tesseract` to extract text.
- 🔄 **Processes all pages** in multi-page PDFs.
- 💾 **Exports each detected table** to its own CSV in the output directory.
- 🧠 **Automatic fallback** mechanism if tables aren't detected.
- 🔁 **Retry logic** and safe temp directory cleanup for stability.

---

## 📦 Installation

### Prerequisites

- Python 3.8+ (tested on Python 3.12.8)
- Tesseract OCR installed and available in your system PATH

### Install dependencies

```bash
pip install -r requirements.txt
```


### 🔧Usage
#### Extract tables from a single PDF:
```bash
python pdf_to_csv_extractor.py input.pdf -o extracted_tables
```
#### Extract tables from all PDFs in a folder:
```bash
python pdf_to_csv_extractor.py ./pdfs_folder -o ./output_folder
```
#### Each extracted table is saved as a CSV file named:
```bash
filename_table_native_p{page}_{table_index}.csv    # For native PDFs
filename_page_{page}_table_{table_index}.csv       # For scanned PDFs
```


![image](https://github.com/user-attachments/assets/f42873aa-9ed3-4573-82f7-7b23a84d17b6)

![image](https://github.com/user-attachments/assets/914e169f-9428-49ba-87cc-939b3c41afc5)

![image](https://github.com/user-attachments/assets/972e0205-f9cd-4c5a-9132-1a3faa1b1999)


Multiple OCRs for efficient extraction
find/search specific terms in extracted text
prints comprehensive summary of OCR results and patterns found
Comprehensive skew detection using multiple methods(# Method 1: Hough Line Transform,  # Method 2: OCR-based structure analysis, # Method 3: Projection Profile, # Method 4: Contour-based detection)
identify special patterns and characters
refined table structure detection (stream and lattice tables)
