import torch
from diffusers import FluxPipeline
import pickle
from tqdm import tqdm

def argparser():
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--captions_pickle_path', type=str, help='Path to the pickle file containing the captions')
    parser.add_argument('--save_path', type=str, help='Path to save the generated images')
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-schnell', help='Model name to use for captioning')

    return parser

if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()

    # Load data from a pickle file
    with open(args.captions_pickle_path, 'rb') as f:
        captions = pickle.load(f)

    pipe = FluxPipeline.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    main_name = args.captions_pickle_path.split('/')[-1].split('.')[0][:-24]

    for i, caption in tqdm(enumerate(captions)):

        image = pipe(
            caption,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

        image.save(f"{args.save_path}/{i}_{main_name}.png")