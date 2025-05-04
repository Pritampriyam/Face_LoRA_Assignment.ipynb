from diffusers import StableDiffusionPipeline
import torch

def generate():
    # 1. Load model
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16
    )
    pipe.unet.load_attn_procs("models/lora_face")
    pipe.to("cuda")
    
    # 2. Generate images
    prompts = [
        ("spacesuit", "sks person in a spacesuit, watercolor painting"),
        ("horse", "sks person riding a horse, watercolor style"),
        ("cricket", "sks person playing cricket, vibrant watercolor")
    ]
    
    os.makedirs("output", exist_ok=True)
    for name, prompt in prompts:
        image = pipe(prompt, num_inference_steps=50).images[0]
        image.save(f"output/{name}.png")
        print(f"Generated: output/{name}.png")

if __name__ == "__main__":
    generate()
