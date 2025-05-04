from PIL import Image
import os

def process_images(input_dir="input", output_size=512):
    os.makedirs(input_dir, exist_ok=True)
    for img_file in os.listdir(input_dir):
        img = Image.open(f"{input_dir}/{img_file}")
        img = img.resize((output_size, output_size))
        img.save(f"{input_dir}/processed_{img_file}")
