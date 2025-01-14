import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import pandas as pd
from pathlib import Path
from torchvision.models import vit_b_16, ViT_B_16_Weights
import warnings
from tqdm import tqdm
import argparse



warnings.filterwarnings('ignore')

class ImageSimilarityCalculator:
    def __init__(self, batch_size):
        # Load pretrained EfficientNetB0 and remove the final classification layer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = ViT_B_16_Weights.DEFAULT
        
        self.model = vit_b_16(weights=weights)
        self.model.heads = nn.Identity()  # Remove classification head
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = weights.transforms()
        self.batch_size = batch_size
    
    def preprocess_cifar_image(self, image_array):
        # Convert numpy array to PIL Image and apply transforms
        image = Image.fromarray(image_array.astype('uint8'))
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def preprocess_file_image(self, image_path):
        # Load and preprocess image from file
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def get_embedding(self, image_tensor):
        embedding = self.model(image_tensor)
        # Normalize the embedding for cosine similarity
        return F.normalize(embedding, p=2, dim=1)
    
    def process_batch(self, cifar_images, file_paths):
        # Process CIFAR images
        cifar_tensors = torch.cat([
            self.preprocess_cifar_image(img) for img in cifar_images
        ])
        cifar_embeddings = self.get_embedding(cifar_tensors)
        
        # Process file images
        file_tensors = torch.cat([
            self.preprocess_file_image(path) for path in file_paths
        ])
        file_embeddings = self.get_embedding(file_tensors)
        
        # Compute similarities using torch's cosine similarity
        similarities = torch.nn.functional.cosine_similarity(
            cifar_embeddings, file_embeddings
        )
        
        return similarities.cpu().numpy()
    
    def compute_similarities(self, cifar_df, cifar_images_dir):
        similarities = {}
        cifar_images_dir = Path(cifar_images_dir)
        
        # Process in batches
        for start_idx in tqdm(range(0, len(cifar_df), self.batch_size), total=len(list(range(0, len(cifar_df), self.batch_size))), desc="Generating similarity scores"):
            end_idx = min(start_idx + self.batch_size, len(cifar_df))
            batch_indices = range(start_idx, end_idx)
            
            # Get batch data
            cifar_images = [cifar_df.iloc[i].image for i in batch_indices]
            file_paths = [
                cifar_images_dir / f"{i}.png" for i in batch_indices
            ]
            
            # Filter out non-existent files
            valid_indices = []
            valid_cifar_images = []
            valid_file_paths = []
            
            for idx, (cifar_img, file_path) in enumerate(zip(cifar_images, file_paths)):
                if file_path.exists():
                    valid_indices.append(batch_indices[idx])
                    valid_cifar_images.append(cifar_img)
                    valid_file_paths.append(file_path)
                else:
                    print(f"Warning: Image {file_path} not found")
            
            if not valid_indices:
                continue
                
            # Process batch
            batch_similarities = self.process_batch(
                valid_cifar_images,
                valid_file_paths
            )
            
            # Store results
            for idx, sim in zip(valid_indices, batch_similarities):
                similarities[idx + 1] = float(sim)  # Convert to Python float
            
            print(f"Processed images {start_idx + 1} to {end_idx}")
                
        return similarities

def argparser():
    parser = argparse.ArgumentParser(description='Arguments for captioning images')
    parser.add_argument('--ref_images_pickle_path', type=str, help='Path to the pickle file containing the reference images')
    parser.add_argument('--gen_images_dir', type=str, help='Path to the pickle file containing the generated images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for  image similarity calculation')
    parser.add_argument(
            '--save_path', 
            default='image_similarities.csv',
            type=str, 
            help="Directory where the similarities will be saved."
        )    
    
    return parser    
# Example usage
def main():
    parser = argparser()
    args = parser.parse_args()
    # Load your CIFAR dataset
    ref_images = pd.read_pickle(args.ref_images_pickle_path)
    
    # Initialize calculator
    calculator = ImageSimilarityCalculator(batch_size=args.batch_size)
    
    # Compute similarities
    similarities = calculator.compute_similarities(
        ref_images,
        args.gen_images_dir
    )
    
    # Convert to DataFrame for easy saving/analysis
    similarity_df = pd.DataFrame.from_dict(
        similarities,
        orient='index',
        columns=['similarity']
    )
    
    # Save results
    similarity_df.to_csv(args.save_path)
    
    print(f"Completed! Results saved to {args.save_path}")

    # Print some statistics
    print("\nSimilarity Statistics:")
    print(f"Average similarity: {similarity_df['similarity'].mean():.4f}")
    print(f"Min similarity: {similarity_df['similarity'].min():.4f}")
    print(f"Max similarity: {similarity_df['similarity'].max():.4f}")

if __name__ == "__main__":
    main()
