import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings('ignore')

class TextImageSimilarityCalculator:
    def __init__(self, batch_size=32, model_name="google/siglip-so400m-patch14-384"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.batch_size = batch_size
        self.model.eval()
    
    def preprocess_cifar_image(self, image_array):
        image = Image.fromarray(image_array.astype('uint8'))
        return self.processor(images=image, return_tensors="pt").to(self.device)
    
    @torch.inference_mode()
    def calculate_similarity(self, images_df, captions):
        similarities = []
        
        for idx in tqdm(range(len(images_df)), desc="Calculating similarities"):
            image = images_df.iloc[idx].image
            caption = captions[idx]
            
            # Process image and text
            image_inputs = self.preprocess_cifar_image(image)
            text_inputs = self.processor(text=[caption], return_tensors="pt").to(self.device)
            
            # Get embeddings
            outputs = self.model(**{
                "pixel_values": image_inputs.pixel_values,
                "input_ids": text_inputs.input_ids,
                "attention_mask": text_inputs.attention_mask
            })
            
            # Calculate similarity score
            logits = outputs.logits_per_image
            prob = torch.sigmoid(logits)
            similarities.append(float(prob[0][0]))
            
        return similarities

def argparser():
    parser = argparse.ArgumentParser(description='Calculate text-image similarity using SigLIP model')
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
        '--model_name',
        type=str,
        default='google/siglip-so400m-patch14-384',
        help='Name of the model to use'
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
    
    calculator = TextImageSimilarityCalculator(batch_size=args.batch_size, model_name=args.model_name)
    
    print("Calculating similarities...")
    similarities = calculator.calculate_similarity(images_df, captions)
    
    # Save results
    results_df = pd.DataFrame({
        'index': range(len(similarities)),
        'similarity': similarities,
        'caption': captions
    })
    results_df.to_csv(args.save_path, index=False)
    
    print(f"\nResults saved to {args.save_path}")
    print("\nSimilarity Statistics:")
    print(f"Average similarity: {sum(similarities)/len(similarities):.4f}")
    print(f"Min similarity: {min(similarities):.4f}")
    print(f"Max similarity: {max(similarities):.4f}")

if __name__ == "__main__":
    main()