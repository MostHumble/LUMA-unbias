import torch
from diffusers import AutoPipelineForText2Image
import pickle
from tqdm import tqdm
import os
import logging
import torch



def argparser():
    """Parses command-line arguments for image generation."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate images from captions using a pretrained diffusion model.")
    parser.add_argument(
        '--captions_pickle_path', 
        type=str, 
        required=True, 
        help="Path to the pickle file containing the captions."
    )
    parser.add_argument(
        '--save_path', 
        type=str, 
        required=True, 
        help="Directory where the generated images will be saved."
    )
    return parser

def setup_logger():
    """Configures the logging settings."""
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def validate_inputs(args):
    """Validates the input paths."""
    if not os.path.isfile(args.captions_pickle_path):
        raise FileNotFoundError(f"Captions file not found: {args.captions_pickle_path}")
    if not args.captions_pickle_path.endswith('.pkl'):
        raise ValueError(f"Expected a pickle file (.pkl), got: {args.captions_pickle_path}")
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        logging.info(f"Created directory: {args.save_path}")

def load_captions(filepath):
    """Loads captions from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def main():
    setup_logger()
    parser = argparser()
    args = parser.parse_args()

    logging.info("Validating inputs...")
    validate_inputs(args)

    logging.info("Loading captions...")
    captions = load_captions(args.captions_pickle_path)
    logging.info(f"Loaded {len(captions)} captions.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    logging.info("Loading model pipeline...")

    pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    generator = torch.Generator("cuda").manual_seed(31)
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")

    logging.info("Starting image generation...")

    for i, caption in tqdm(enumerate(captions), total=len(captions), desc="Generating Images"):
        try:
            image = pipe(
                prompt = caption,
                guidance_scale=3.5,
                generator=generator
            ).images[0]

            save_path = os.path.join(args.save_path, f"{i}.png")
            image.save(save_path)
        except Exception as e:
            logging.error(f"Error processing caption {i}: {e}")

    logging.info("Image generation completed.")

if __name__ == "__main__":
    main()
