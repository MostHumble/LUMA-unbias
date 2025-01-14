import torch
from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoProcessor
from huggingface_hub import snapshot_download
from transformers.dynamic_module_utils import get_imports
import time
from tqdm import tqdm
import pandas 
import pickle

torch.set_float32_matmul_precision("high")

# Configuration options
OVERWRITE = True  # Boolean option to allow overwriting existing caption files
PREPEND_STRING = ""  # Prefix string to prepend to the generated caption
APPEND_STRING = ""  # Suffix string to append to the generated caption
BATCH_SIZE = 7 # How many images to process at one time. A 24gb VRAM 3090 can handle 7. A 6gb VRAM GPU can handle a batch size of 1.
PRINT_PROCESSING_STATUS = False  # Option to print processing status of images
PRINT_CAPTIONS = False  # Option to print captions to the console
DETAIL_MODE = 3 # The level of verbosity for the output caption.

print(f"Captioning with batch size: {BATCH_SIZE}")

def fixed_get_imports(filename: str | Path) -> List[str]:
    imports = get_imports(filename)
    return [imp for imp in imports if imp != "flash_attn"] if str(filename).endswith("modeling_florence2.py") else imports

def download_and_load_model(model_name: str) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device available: {device}')

    model_path = Path("models") / model_name.replace('/', '_')
    if not model_path.exists():
        print(f"Downloading {model_name} model to: {model_path}")
        snapshot_download(repo_id=model_name, local_dir=model_path, local_dir_use_symlinks=False)

    print(f"Loading model {model_name}...")
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("Model loaded.")
    model = torch.compile(model, mode="reduce-overhead")
    return model, processor

def run_model_batch(images: torch.Tensor, model: AutoModelForCausalLM, processor: AutoProcessor,
                     num_beams: int = 3, max_new_tokens: int = 512) -> List[str]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prompt = '<MORE_DETAILED_CAPTION>'

    inputs = processor(text=[prompt]*images.size(0), images=images, return_tensors="pt", do_rescale=False).to(device)

    # Keep input_ids as Long type and only convert pixel_values to bfloat16
    inputs["input_ids"] = inputs["input_ids"]
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=num_beams,
    )

    results = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return results


def process_images_recursive(images: pandas.DataFrame,
                            model: AutoModelForCausalLM,
                            processor: AutoProcessor, 
                            batch_size: int = 8,
                            save_name='test') -> Tuple[int, float]:
    import numpy as np

    start_time = time.time()
    total_images = 0
    all_captions = []
    num_batches = len(images) // batch_size + (1 if len(images) % batch_size > 0 else 0)

    for i in tqdm(range(num_batches), desc="Processing batches"):
        # Extract and prepare the batch of images
        try:
            img_batch_np = np.stack(images.iloc[i * batch_size:(i + 1) * batch_size]['image'].values)  # Stack into a single array
            img_batch = torch.tensor(img_batch_np, dtype=torch.float32)  # Convert to PyTorch tensor
        except Exception as e:
            print(f"Error preparing batch {i}: {e}")
            continue  # Skip the batch and move to the next
        try:
            captions = run_model_batch(img_batch, model, processor)
            all_captions.extend(captions)
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            with open(f"/home/sklioui/captions/captions_partial_{save_name}.pkl", "wb") as f:
                pickle.dump(all_captions, f)

        total_images += len(img_batch)
    
    # Save all captions to a pickle file
    with open(f"/home/sklioui/captions/captions_{save_name}.pkl", "wb") as f:
        pickle.dump(all_captions, f)

    total_time = time.time() - start_time
    return total_images, total_time

def argparser():
    import argparse
    parser = argparse.ArgumentParser(description='Arguments for captioning images')
    parser.add_argument('--images_pickle_path', type=str, help='Path to the pickle file containing the images')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for captioning images')
    parser.add_argument('--model_name', type=str, default='microsoft/Florence-2-large', help='Model name to use for captioning')

    return parser

def main():
    parser = argparser()
    args = parser.parse_args()

    # Main execution
    model, processor = download_and_load_model(args.model_name)

    # Load the images from the pickle file
    with open(args.images_pickle_path, "rb") as f:
        images = pickle.load(f)
    
    print(f"Loaded {len(images)} images from {args.images_pickle_path}")
    print(images)

    total_images, total_time = process_images_recursive(images,
                                                        model,
                                                        processor,
                                                        batch_size=args.batch_size,
                                                        save_name=args.images_pickle_path.split('/')[-1].split('.')[0])

    print(f"Total images captioned: {total_images}")
    print(f"Total time taken: {total_time:.2f} seconds")

    # Fix for divide-by-zero when calculating average time per image
    if total_images > 0:
        print(f"Average time per image: {total_time / total_images:.2f} seconds")
    else:
        print("No images were processed, so no average time to display.")

if __name__ == "__main__":
    main()