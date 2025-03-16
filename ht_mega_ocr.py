import pytesseract
from tqdm import tqdm
import os
import re
from PIL import Image

def extract_page_and_part(filename):

    page_match = re.search(r"Sayfa\s+(\d+)", filename, re.IGNORECASE)
    if page_match:
        page_num = int(page_match.group(1))
    else:
        page_num = 1
    part_match = re.search(r'_part(\d+)', filename, re.IGNORECASE)
    if part_match:
        part_num = int(part_match.group(1))
    else:
        part_num = 1
    return page_num, part_num

def ocr_images(image_folder, output_file, lang="tur", psm=3, oem=3):

    tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    if not image_files:
        print("No image files found in the folder.")
        return

    print(f"Found {len(image_files)} images. Starting OCR...")
    pages = {}
    config = f"--oem {oem} --psm {psm}"

    for image_file in tqdm(image_files, desc="Processing Images", unit="image"):
        image_path = os.path.join(image_folder, image_file)
        page_num, part_num = extract_page_and_part(image_file)
        try:
            text = pytesseract.image_to_string(Image.open(image_path), lang=lang, config=config)
            text = text.strip()
            if page_num not in pages:
                pages[page_num] = {}
            pages[page_num][part_num] = text
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        for page_num in sorted(pages.keys()):
            part_texts = [pages[page_num][p] for p in sorted(pages[page_num].keys())]
            combined_text = "\n".join(part_texts)

            f.write(f"\n--- Sayfa {page_num} ---\n")
            f.write(combined_text + "\n")

    print(f"OCR complete! Text saved to: {output_file}")

image_folder_path = r"x"
output_text_path = r"xx"

ocr_images(image_folder_path, output_text_path, lang="tur", psm=3, oem=3)
