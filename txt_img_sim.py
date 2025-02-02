import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse
import warnings
import numpy as np

warnings.simplefilter("ignore")

class TextImageScoresCalculator:
    def __init__(self, batch_size=32, model_id="zer0int/LongCLIP-L-Diffusers", maxtokens=248):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # Configure and load the model
        config = CLIPConfig.from_pretrained(model_id)
        config.text_config.max_position_embeddings = maxtokens
        
        self.model = CLIPModel.from_pretrained(
            model_id, 
            torch_dtype=self.dtype,
            config=config
        ).to(self.device)
        
        self.processor = CLIPProcessor.from_pretrained(
            model_id,
            padding="max_length",
            max_length=maxtokens,
            return_tensors="pt",
            truncation=True
        )
        
        self.batch_size = batch_size
        self.model.eval()
    
    def preprocess_cifar_image(self, image_array):
        image_tensor = torch.from_numpy(image_array)
        final_image = image_tensor.unsqueeze(0).permute(0, 3, 1, 2)
        return final_image
    
    @torch.inference_mode()
    def calculate_scores(self, images_df, captions):
        scores = []
        
        for idx in tqdm(range(len(images_df)), desc="Calculating similarities"):
            image = images_df.iloc[idx].image
            caption = captions[idx]
            
            # Process image and text
            final_image = self.preprocess_cifar_image(image)
            inputs = self.processor(
                text=caption,
                images=final_image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get similarity scores
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image
            scores.append(float(logits[0]))
            
        return scores

def argparser():
    parser = argparse.ArgumentParser(description='Calculate text-image similarity using a CLIP model')
    parser.add_argument(
        '--images_pickle_path',
        type=str,
        default='/home/sklioui/cifar100_man_woman_baby_girl_boy.pkl',
        help='Path to the pickle file containing CIFAR images'
    )
    parser.add_argument(
        '--captions_pickle_path',
        type=str,
        default='/home/sklioui/captions/captions_cifar100_man_woman_baby_girl_boy.pkl',
        help='Path to the pickle file containing generated captions'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='text_image_similarities.csv',
        help='Path to save the similarity scores'
    )
    parser.add_argument(
        '--model_id',
        type=str,
        default='zer0int/LongCLIP-L-Diffusers',
        help='Name of the model to use'
    )
    parser.add_argument(
        '--maxtokens',
        type=int,
        default=248,
        help='Maximum number of tokens for the model'
    )
    return parser

def main():
    parser = argparser()
    args = parser.parse_args()
    
    print("Loading images and captions...")
    images_df = pd.read_pickle(args.images_pickle_path)
    with open(args.captions_pickle_path, 'rb') as f:
        captions = pickle.load(f)
    
    print(f"Loaded {len(images_df)} images and {len(captions)} captions")
    
    calculator = TextImageScoresCalculator(
        batch_size=args.batch_size,
        model_id=args.model_id,
        maxtokens=args.maxtokens
    )
    
    print("Calculating scores...")
    scores = calculator.calculate_scores(images_df, captions)
    
    # Save results
    results_df = pd.DataFrame({
        'index': range(len(scores)),
        'score': scores,
        'caption': captions
    })
    results_df.to_csv(args.save_path, index=False)
    
    print(f"\nResults saved to {args.save_path}")
    print("\nSimilarity Statistics:")
    print(f"Average score: {sum(scores)/len(scores):.4f}")
    print(f"Min score: {min(scores):.4f}")
    print(f"Max score: {max(scores):.4f}")

if __name__ == "__main__":
    main()