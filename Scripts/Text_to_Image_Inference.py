import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import DiffusionPipeline


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger() -> None:
    """Configure logging format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ---------------------------------------------------------------------
# Main inference logic
# ---------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    setup_logger()
    set_seed(args.seed)

    logging.info("Loading SDXL base model...")
    pipe = DiffusionPipeline.from_pretrained(
        args.model_name,
        variant="fp16",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")

    pipe.enable_model_cpu_offload()
    logging.info("Base model loaded")

    logging.info("Loading LoRA weights...")
    pipe.load_lora_weights(args.lora_path)
    logging.info("LoRA weights loaded")

    logging.info("Loading test dataset...")
    with open(args.test_json, "r") as f:
        test_set = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Starting image generation...")
    for idx, sample in enumerate(test_set):
        prompt = sample["text"]
        image_name = sample.get("image", f"{idx}.png")

        image = pipe(
            prompt=prompt,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
        ).images[0]

        image.save(output_dir / image_name)

        if idx % args.log_interval == 0:
            logging.info(f"Generated {idx} images")
            logging.debug(f"Prompt: {prompt}")

    logging.info("Inference completed successfully.")


# ---------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDXL + LoRA Text-to-Image Inference")

    parser.add_argument(
        "--model_name",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base SDXL model name or path",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA checkpoint",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        required=True,
        help="Path to test JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save generated images",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--log_interval", type=int, default=200)

    args = parser.parse_args()
    main(args)
