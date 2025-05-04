import os
from PIL import Image

def validate_images(input_dir):
    """Check if images exist and are valid"""
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} not found")
        
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(images) < 3:
        raise ValueError("At least 3 images required in input/ folder")
    
    print(f"Found {len(images)} valid images")

def create_folders():
    """Ensure all folders exist"""
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("models", exist_ok=True)
