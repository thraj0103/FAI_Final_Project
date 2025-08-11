# src/models_and_attacks.py

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm
import json
from collections import defaultdict
import time

# Define critical classes for navigation (COCO dataset indices)
CRITICAL_CLASSES = {
    0: 'person',
    1: 'bicycle', 
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

class SimpleAdversarialAttacks:
    """Simple adversarial attacks using random perturbations"""
    
    def __init__(self, model):
        self.model = model
        self.critical_classes = CRITICAL_CLASSES
    
    def random_attack(self, image, epsilon=0.03):
        """Random noise attack"""
        noise = np.random.uniform(-epsilon, epsilon, image.shape)
        perturbed = np.clip(image + noise, 0, 1)
        return perturbed
    
    def targeted_attack(self, image, epsilon=0.03, num_tries=10):
        """
        Try multiple random perturbations and keep the one that reduces detections most
        """
        original = image.copy()
        
        # Get original detections
        orig_results = self.model((original * 255).astype(np.uint8), verbose=False)
        orig_count = 0
        if orig_results[0].boxes is not None:
            classes = orig_results[0].boxes.cls.cpu().numpy().astype(int)
            orig_count = sum(1 for cls in classes if cls in self.critical_classes)
        
        best_perturbed = original
        min_detections = orig_count
        
        # Try multiple random perturbations
        for _ in range(num_tries):
            noise = np.random.uniform(-epsilon, epsilon, original.shape)
            perturbed = np.clip(original + noise, 0, 1)
            
            # Test detection
            results = self.model((perturbed * 255).astype(np.uint8), verbose=False)
            count = 0
            if results[0].boxes is not None:
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                count = sum(1 for cls in classes if cls in self.critical_classes)
            
            # Keep if better
            if count < min_detections:
                min_detections = count
                best_perturbed = perturbed
        
        return best_perturbed


def load_yolo_model(model_path='yolov8n.pt', device='cpu'):
    """Load YOLOv8 model"""
    print("Loading YOLOv8...")
    model = YOLO(model_path)
    print(f"Loaded YOLOv8 nano on {device}")
    return model


def visualize_detections(image, model, ax=None, title="YOLOv8 Detections"):
    """Visualize YOLO detections on image"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Run detection
    results = model((image * 255).astype(np.uint8), verbose=False)
    
    # Display image
    ax.imshow(image)
    
    # Draw boxes
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        
        critical_count = 0
        for box, cls, conf in zip(boxes, classes, confidences):
            if cls in CRITICAL_CLASSES:
                critical_count += 1
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1-5, f"{CRITICAL_CLASSES[cls]} {conf:.2f}", 
                       color='red', fontsize=10, weight='bold',
                       bbox=dict(facecolor='white', alpha=0.8))
    
    ax.set_title(title)
    ax.axis('off')
    
    return ax


def evaluate_attacks(dataset, attacker, num_samples=50, epsilons=[0.01, 0.04, 0.07, 0.1]):
    """Evaluate attack effectiveness"""
    results = defaultdict(list)
    
    print(f"Evaluating attacks on {num_samples} images...")
    
    for img_idx in tqdm(range(num_samples)):
        img, _ = dataset.load_image(img_idx)
        
        # Original detection
        orig_results = attacker.model((img * 255).astype(np.uint8), verbose=False)
        orig_critical = 0
        if orig_results[0].boxes is not None:
            classes = orig_results[0].boxes.cls.cpu().numpy().astype(int)
            orig_critical = sum(1 for cls in classes if cls in CRITICAL_CLASSES)
        
        # Skip if no critical objects
        if orig_critical == 0:
            continue
            
        # Test each epsilon
        for eps in epsilons:
            # Attack
            adv_img = attacker.targeted_attack(img, epsilon=eps, num_tries=10)
            adv_results = attacker.model((adv_img * 255).astype(np.uint8), verbose=False)
            
            adv_critical = 0
            if adv_results[0].boxes is not None:
                classes = adv_results[0].boxes.cls.cpu().numpy().astype(int)
                adv_critical = sum(1 for cls in classes if cls in CRITICAL_CLASSES)
            
            results[eps].append({
                'original': orig_critical,
                'adversarial': adv_critical,
                'suppressed': orig_critical - adv_critical
            })
    
    return results


def plot_attack_results(evaluation_results, save_path=None):
    """Plot attack effectiveness results"""
    epsilons = sorted(evaluation_results.keys())
    suppression_rates = []
    avg_detected = []

    for eps in epsilons:
        data = evaluation_results[eps]
        if data:
            total_orig = sum(d['original'] for d in data)
            total_supp = sum(d['suppressed'] for d in data)
            suppression_rates.append(total_supp / total_orig * 100)
            avg_detected.append(np.mean([d['adversarial'] for d in data]))
        else:
            suppression_rates.append(0)
            avg_detected.append(0)

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Suppression rate
    ax1.plot(epsilons, suppression_rates, 'o-', linewidth=2, markersize=8, color='red')
    ax1.set_xlabel('Epsilon', fontsize=12)
    ax1.set_ylabel('Suppression Rate (%)', fontsize=12)
    ax1.set_title('Critical Object Suppression', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Average detections
    ax2.bar(range(len(epsilons)), avg_detected, color='blue', alpha=0.7)
    ax2.set_xlabel('Epsilon', fontsize=12)
    ax2.set_ylabel('Avg Objects Detected', fontsize=12)
    ax2.set_title('Detection Performance', fontsize=14)
    ax2.set_xticks(range(len(epsilons)))
    ax2.set_xticklabels(epsilons)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    plt.show()
    
    return fig


def print_attack_summary(evaluation_results):
    """Print attack effectiveness summary"""
    print("ATTACK EFFECTIVENESS SUMMARY")
    print("=" * 50)

    for eps in sorted(evaluation_results.keys()):
        data = evaluation_results[eps]
        if data:
            total_original = sum(d['original'] for d in data)
            total_adversarial = sum(d['adversarial'] for d in data)
            total_suppressed = sum(d['suppressed'] for d in data)
            suppression_rate = (total_suppressed / total_original) * 100
            
            print(f"\nε = {eps}:")
            print(f"  Images evaluated: {len(data)}")
            print(f"  Total objects (original): {total_original}")
            print(f"  Total objects (adversarial): {total_adversarial}")
            print(f"  Objects suppressed: {total_suppressed}")
            print(f"  Suppression rate: {suppression_rate:.1f}%")


# Test function to verify imports work correctly
def test_imports():
    """Test that all imports and classes work correctly"""
    print("Testing models_and_attacks module...")
    print(f"Critical classes defined: {list(CRITICAL_CLASSES.values())}")
    print("SimpleAdversarialAttacks class available: ✓")
    print("All functions available: ✓")
    return True


# Only run if script is executed directly (not imported)
if __name__ == "__main__":
    test_imports()