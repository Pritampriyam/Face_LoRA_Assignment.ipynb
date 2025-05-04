import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
import os

def train():
    # 1. Setup
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Load Model
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    
    # 3. Configure LoRA
    lora_config = LoraConfig(
        r=4,
        lora_alpha=32,
        target_modules=["to_k", "to_q", "to_v"],
        lora_dropout=0.05,
    )
    unet = get_peft_model(unet, lora_config)
    unet.to(device)
    
    # 4. Training Loop (simplified)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
    for epoch in range(30):
        # Add your training logic here
        print(f"Epoch {epoch+1}")
        
    # 5. Save
    os.makedirs("models", exist_ok=True)
    unet.save_pretrained("models/lora_face")
    print("Training complete! Model saved to models/lora_face")

if __name__ == "__main__":
    train()
