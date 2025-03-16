import os
from PyPDF2 import PdfReader

def extract_pdf_text(pdf_path, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    reader = PdfReader(pdf_path)
    with open(output_file, "w", encoding="utf-8") as txt_file:
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                txt_file.write(f"\n--- Page {page_num} ---\n")
                txt_file.write(text.strip() + "\n")
            else:
                txt_file.write(f"\n--- Page {page_num}: No text found ---\n")
    print(f"Text extracted and saved to: {output_file}")

if __name__ == "__main__":
    pdf_path = r"xx"
    output_file = r"xx"

    extract_pdf_text(pdf_path, output_file)
