# Stable Diffusion LoRA Face Project

Generate watercolor art of a person using LoRA fine-tuning.

## 🚀 Quick Start
```bash
git clone https://github.com/your-username/stable-diffusion-lora.git
cd stable-diffusion-lora
pip install -r requirements.txt

# Train LoRA
python train_lora.py --input_dir input/ --output_dir models/

# Generate images
python generate.py --lora_path models/ --prompt "watercolor painting of sks person riding a horse"
```
## 📂 File Structure
- `input/`: Face images for training
- `output/`: Generated artwork
- `train_lora.py`: Training script
- `generate.py`: Image generation
