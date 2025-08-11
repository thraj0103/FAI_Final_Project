# src/dataset_preparation.py

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


class SimpleNavigationDataset:
    """Dataset class for loading nuScenes images"""
    
    def __init__(self, num_images=400, data_root='data/v1.0-mini'):
        self.samples_dir = os.path.join(data_root, 'samples/CAM_FRONT')
        self.target_size = (640, 640)
        
        if not os.path.exists(self.samples_dir):
            raise FileNotFoundError(f"CAM_FRONT folder not found at {self.samples_dir}")
        
        all_images = sorted([f for f in os.listdir(self.samples_dir) if f.endswith('.jpg')])
        self.images = all_images[:min(num_images, len(all_images))]
        
        if len(self.images) == 0:
            raise ValueError("No images found in the dataset directory")
            
        print(f"Dataset ready with {len(self.images)} images")
    
    def load_image(self, idx):
        """Load and resize image"""
        if idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.images)} images")
            
        img_path = os.path.join(self.samples_dir, self.images[idx])
        img = cv2.imread(img_path)
        
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)
        return img / 255.0, img_path  # Normalize to [0,1]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Make dataset indexable"""
        return self.load_image(idx)


def verify_dataset(data_root='data/v1.0-mini'):
    """Verify nuScenes dataset installation"""
    samples_dir = os.path.join(data_root, 'samples/CAM_FRONT')
    
    if os.path.exists(samples_dir):
        images = [f for f in os.listdir(samples_dir) if f.endswith('.jpg')]
        print(f"✅ Found {len(images)} front camera images")
        return True, len(images)
    else:
        print(f"❌ CAM_FRONT folder not found at {samples_dir}")
        return False, 0


def visualize_samples(dataset, num_samples=6, save_path=None):
    """Visualize sample images from dataset"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Calculate sample interval
    interval = len(dataset) // num_samples if len(dataset) > num_samples else 1
    
    for i in range(min(num_samples, len(dataset))):
        idx = i * interval
        img, path = dataset.load_image(idx)
        axes[i].imshow(img)
        axes[i].set_title(f"Image {idx}")
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(dataset), num_samples):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device():
    """Get the best available device"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    return device


def create_dataset(num_images=400, data_root='data/v1.0-mini'):
    """Convenience function to create and verify dataset"""
    # Verify dataset exists
    exists, total_images = verify_dataset(data_root)
    
    if not exists:
        raise FileNotFoundError(f"Dataset not found at {data_root}")
    
    # Create dataset
    dataset = SimpleNavigationDataset(num_images=num_images, data_root=data_root)
    
    return dataset


# Test function to verify module works correctly
def test_module():
    """Test that all functions work correctly"""
    print("Testing dataset_preparation module...")
    
    # Test device detection
    device = get_device()
    
    # Test seed setting
    set_random_seeds(42)
    
    print("Module functions available: ✓")
    print("SimpleNavigationDataset class available: ✓")
    
    return True


# Only run if script is executed directly
if __name__ == "__main__":
    # Example usage
    print("Running dataset_preparation module test...")
    
    # Set seeds
    set_random_seeds(42)
    
    # Get device
    device = get_device()
    
    # Try to create dataset
    try:
        dataset = create_dataset(num_images=10)
        print(f"Successfully created dataset with {len(dataset)} images")
        
        # Visualize samples
        visualize_samples(dataset, num_samples=6)
        
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("Please ensure nuScenes mini dataset is downloaded and extracted to 'data/v1.0-mini'")