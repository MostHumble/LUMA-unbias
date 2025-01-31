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
import numpy as np
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms


warnings.filterwarnings('ignore')

class ImageSimilarityCalculator:
    def __init__(self, batch_size, metric='cosine', net_type='alex', normalize=True):
        # Load pretrained EfficientNetB0 and remove the final classification layer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metric = metric
        if metric == 'cosine':
            weights = ViT_B_16_Weights.DEFAULT
            self.model = vit_b_16(weights=weights)
            self.model.heads = nn.Identity()  # Remove classification head
            self.model = self.model.to(self.device)
            self.model.eval()
            self.transform = weights.transforms()
        if metric == 'lpips':
            self.normalize = normalize
            self.net_type = net_type
            self.model = LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=normalize).to(self.device)
            if normalize:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x / 255.0)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: 2 * x / 255.0 - 1)
                ])
        self.batch_size = batch_size
    
    def _is_path(self, image):
        return isinstance(image, (str, Path))

    def _is_array(self, image):
        return isinstance(image, np.ndarray)

    def preprocess_image(self, image):
        if self._is_path(image):
            img = Image.open(image)
            if img.mode != 'RGB':
                img = img.convert('RGB')
        elif self._is_array(image):
            img = Image.fromarray(image.astype('uint8'))
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        return self.transform(img).unsqueeze(0).to(self.device)
    
    @torch.inference_mode()
    def get_embedding(self, image_tensor):
        embedding = self.model(image_tensor)
        # Normalize the embedding for cosine similarity
        return F.normalize(embedding, p=2, dim=1)
    
    def process_batch(self, source_images, target_images):
        # Process source images

        if self.metric == 'cosine':
            source_tensors = torch.cat([
            self.preprocess_image(img) for img in source_images
            ])
            
            # Process target images
            target_tensors = torch.cat([
                self.preprocess_image(img) for img in target_images
            ])
            source_embeddings = self.get_embedding(source_tensors)
            target_embeddings = self.get_embedding(target_tensors)
            
            # Compute similarities using torch's cosine similarity
            similarities = torch.nn.functional.cosine_similarity(
                source_embeddings, target_embeddings
            )
            
            return similarities.cpu().numpy()
        
        if self.metric == 'lpips':
            similarities = []
            for image_source, image_target in zip(source_images, target_images):
                similarity = self.model(self.preprocess_image(image_source), self.preprocess_image(image_target))
                similarities.append(similarity)

            similarities = torch.stack(similarities)
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
    
    def compute_class_similarities(self, images_df, max_samples=1000):
        results = {}
        labels = images_df['label'].unique()
        
        for label in tqdm(labels, desc="Processing classes"):
            # Get images for current class
            class_images = images_df[images_df['label'] == label]
            other_images = images_df[images_df['label'] != label]
            
            # Sample pairs for in-class similarity
            in_class_pairs = []
            n_images = len(class_images)
            if n_images > 1:
                idx1 = np.random.randint(0, n_images, min(max_samples, n_images))
                idx2 = np.random.randint(0, n_images, min(max_samples, n_images))
                mask = idx1 != idx2
                in_class_pairs = list(zip(
                    class_images.iloc[idx1[mask]].image.values,
                    class_images.iloc[idx2[mask]].image.values
                ))
            
            # Sample pairs for out-class similarity
            out_class_pairs = []
            if len(other_images) > 0:
                idx_class = np.random.randint(0, len(class_images), max_samples)
                idx_other = np.random.randint(0, len(other_images), max_samples)
                out_class_pairs = list(zip(
                    class_images.iloc[idx_class].image.values,
                    other_images.iloc[idx_other].image.values
                ))
            
            # Process pairs in batches
            in_class_similarities = []
            out_class_similarities = []
            
            # Process in-class similarities
            for i in range(0, len(in_class_pairs), self.batch_size):
                batch = in_class_pairs[i:i + self.batch_size]
                if batch:
                    similarities = self.process_batch(
                        [p[0] for p in batch],
                        [p[1] for p in batch]
                    )
                    in_class_similarities.extend(similarities)
            
            # Process out-class similarities
            for i in range(0, len(out_class_pairs), self.batch_size):
                batch = out_class_pairs[i:i + self.batch_size]
                if batch:
                    similarities = self.process_batch(
                        [p[0] for p in batch],
                        [p[1] for p in batch]
                    )
                    out_class_similarities.extend(similarities)
            
            # Store results
            results[label] = {
                'in_class': {
                    'mean': np.mean(in_class_similarities) if in_class_similarities else None,
                    'std': np.std(in_class_similarities) if in_class_similarities else None,
                    'n_comparisons': len(in_class_similarities)
                },
                'out_class': {
                    'mean': np.mean(out_class_similarities) if out_class_similarities else None,
                    'std': np.std(out_class_similarities) if out_class_similarities else None,
                    'n_comparisons': len(out_class_similarities)
                }
            }
        
        return results

def argparser():
    parser = argparse.ArgumentParser(description='Arguments for captioning images')
    parser.add_argument('--ref_images_pickle_path', type=str, help='Path to the pickle file containing the reference images')
    parser.add_argument('--gen_images_dir', type=str, help='Path to the folder containing the generated images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for  image similarity calculation')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum number of samples to use for similarity calculation')
    parser.add_argument(
            '--save_path', 
            default='image_similarities.csv',
            type=str, 
            help="Directory where the similarities will be saved."
        )    
    parser.add_argument('--metric', type=str, default='cosine', help='Similarity metric to use (cosine or lpips)')
    parser.add_argument('--net_type', type=str, default='alex', help='Network type for lpips metric')
    parser.add_argument('--normalize', type=bool, default=True, help='Normalize the images for lpips metric')
    
    return parser    

def get_suffix(args):
    if args.metric != 'cosine':
        return f"_{args.metric}_{args.net_type}"
    return ""
    
def main():
    parser = argparser()
    args = parser.parse_args()
    # Load your CIFAR dataset
    ref_images = pd.read_pickle(args.ref_images_pickle_path)
    
    # Initialize calculator
    calculator = ImageSimilarityCalculator(batch_size=args.batch_size, metric=args.metric, net_type=args.net_type, normalize=args.normalize)

    if not args.gen_images_dir:
        # Compute class similarities
        results = calculator.compute_class_similarities(ref_images, args.max_samples)

        # Convert to DataFrame
        in_class_df = pd.DataFrame({k: v['in_class'] for k, v in results.items()}).T
        out_class_df = pd.DataFrame({k: v['out_class'] for k, v in results.items()}).T
        
        # Save results
        in_class_df.to_csv(f'in_class_similarities{get_suffix(args)}.csv')
        out_class_df.to_csv(f'out_class_similarities{get_suffix(args)}.csv')
        
        print("\nIn-class similarity statistics:")
        print(in_class_df.describe())
        print("\nOut-class similarity statistics:")
        print(out_class_df.describe())
    
    else:
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
