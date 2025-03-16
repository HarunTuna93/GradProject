import os
from PIL import Image, ImageChops


def remove_black_borders(input_path, output_path):
    img = Image.open(input_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    bg = Image.new("RGB", img.size, (0, 0, 0))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        cropped_img = img.crop(bbox)
        cropped_img.save(output_path)
    else:
        img.save(output_path)


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if os.path.isdir(input_path):
            continue
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            continue
        output_path = os.path.join(output_folder, filename)
        remove_black_borders(input_path, output_path)
        print(f"Processed {filename}")


input_folder = r"x"
output_folder = r"xx"

process_folder(input_folder, output_folder)
