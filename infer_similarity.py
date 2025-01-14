import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from torch.nn.functional import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class ImageSimilarityCalculator:
    def __init__(self, batch_size=32):
        # Load pretrained EfficientNetB0 and remove the final classification layer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.classifier = nn.Identity()  # Remove the final layer for embeddings
        self.model = self.model.to(self.device).eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # EfficientNet expects 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.batch_size = batch_size
    
    def preprocess_cifar_images(self, cifar_images):
        # Convert CIFAR images (numpy arrays) to tensors
        tensors = [self.transform(Image.fromarray(img.astype('uint8'))) for img in cifar_images]
        return torch.stack(tensors).to(self.device)
    
    def preprocess_file_images(self, file_paths):
        # Load and preprocess images from file paths
        tensors = []
        for path in file_paths:
            image = Image.open(path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            tensors.append(self.transform(image))
        return torch.stack(tensors).to(self.device)
    
    def get_embeddings(self, image_tensors):
        with torch.no_grad():
            embeddings = self.model(image_tensors)
        return embeddings
    
    def compute_similarities(self, cifar_df, cifar_images_dir):
        similarities = {}
        cifar_images_dir = Path(cifar_images_dir)
        
        # Preprocess all CIFAR images
        cifar_tensors = self.preprocess_cifar_images(cifar_df.image.values)
        cifar_embeddings = self.get_embeddings(cifar_tensors)
        
        # Process file images in batches
        file_paths = sorted(cifar_images_dir.glob("*.jpg"))
        for batch_start in range(0, len(file_paths), self.batch_size):
            batch_paths = file_paths[batch_start:batch_start + self.batch_size]
            file_tensors = self.preprocess_file_images(batch_paths)
            file_embeddings = self.get_embeddings(file_tensors)
            
            # Compute similarities between CIFAR and batch embeddings
            for idx, file_embedding in enumerate(file_embeddings):
                similarity = cosine_similarity(
                    cifar_embeddings,
                    file_embedding.unsqueeze(0).expand_as(cifar_embeddings)
                ).mean().item()
                image_index = int(batch_paths[idx].stem)  # Extract index from filename
                similarities[image_index] = similarity
            
            print(f"Processed batch {batch_start // self.batch_size + 1}")
        
        return similarities

# Example usage
def main():
    # Load your CIFAR dataset
    cifar_df = pd.read_pickle('cifar100_man_woman_baby_girl_boy.pkl')
    
    # Initialize calculator
    calculator = ImageSimilarityCalculator(batch_size=32)
    
    # Compute similarities
    similarities = calculator.compute_similarities(
        cifar_df,
        'images/images_cifar'
    )
    
    # Convert to DataFrame for easy saving/analysis
    similarity_df = pd.DataFrame.from_dict(
        similarities,
        orient='index',
        columns=['similarity']
    )
    
    # Save results
    similarity_df.to_csv('image_similarities.csv')
    
    print("Completed! Results saved to image_similarities.csv")

if __name__ == "__main__":
    main()
