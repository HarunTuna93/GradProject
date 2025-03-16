import os
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path

def split_pdf(input_pdf_path, output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)
    pdf_reader = PdfReader(input_pdf_path)
    total_pages = len(pdf_reader.pages)
    print(f"Total pages in PDF: {total_pages}")
    individual_pdfs = []
    for i in range(total_pages):
        pdf_writer = PdfWriter()
        pdf_writer.add_page(pdf_reader.pages[i])
        output_pdf_path = os.path.join(output_folder_path, f"page_{i + 1}.pdf")
        with open(output_pdf_path, 'wb') as output_pdf:
            pdf_writer.write(output_pdf)
        individual_pdfs.append(output_pdf_path)
        print(f"Saved page {i + 1} as {output_pdf_path}.")
    return individual_pdfs


def pdf_pages_to_images(pdf_pages, output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)
    for page_number, pdf_path in enumerate(pdf_pages, start=1):
        images = convert_from_path(pdf_path, dpi=300)
        for image in images:
            output_image_path = os.path.join(output_folder_path, f"page_{page_number}.jpeg")
            image.save(output_image_path, 'JPEG')
            print(f"Converted {pdf_path} to {output_image_path}.")


if __name__ == "__main__":
    input_pdf = r"x"
    split_folder = r"xx"
    image_folder = r"xxx"

    pdf_pages = split_pdf(input_pdf, split_folder)
    pdf_pages_to_images(pdf_pages, image_folder)
