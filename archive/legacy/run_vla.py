import argparse
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np
import os

def run_inference(model_path, prompt_text, image_path):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_path} on {DEVICE}...")

    # 1. Load Processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # 2. Load Model with 4-bit Quantization
    vla = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # 3. Prepare Inputs
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
        print(f"Loaded image from {image_path}")
    else:
        print("Using dummy red image (no image path provided or file not found).")
        image = Image.new('RGB', (224, 224), color='red') 

    # OpenVLA expects a specific prompt format: "In: <instruction>\nOut:"
    formatted_prompt = f"In: {prompt_text}\nOut:"
    print(f"Using prompt: {formatted_prompt}")

    # 4. Process Inputs
    inputs = processor(formatted_prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # 5. Run Inference
    print("Predicting action...")
    with torch.inference_mode():
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    # 6. Output Result
    print(f"\nPredicted Action Vector (7-DoF):\n{action}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenVLA inference with custom inputs.")
    parser.add_argument("--model_path", type=str, default="openvla/openvla-7b", help="Path to the model or Hugging Face ID.")
    parser.add_argument("--prompt", type=str, default="Pick up the red object", help="The text instruction for the robot.")
    parser.add_argument("--image_path", type=str, help="Path to the input image file.")

    args = parser.parse_args()

    run_inference(args.model_path, args.prompt, args.image_path)