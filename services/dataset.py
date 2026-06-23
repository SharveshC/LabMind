import os
import glob
from core.schemas import DatasetProfile
from PIL import Image

class DatasetService:
    def analyze(self, dataset_path: str) -> DatasetProfile:
        """Analyzes an image classification dataset."""
        classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        num_classes = len(classes)
        
        class_distribution = {}
        total_samples = 0
        corrupted = 0
        
        # We'll sample up to 10 images to estimate avg width/height for speed
        sampled_widths = []
        sampled_heights = []
        
        for cls in classes:
            cls_path = os.path.join(dataset_path, cls)
            # Find all common image formats
            images = glob.glob(os.path.join(cls_path, "*.jpg")) + \
                     glob.glob(os.path.join(cls_path, "*.png")) + \
                     glob.glob(os.path.join(cls_path, "*.jpeg"))
            
            count = len(images)
            class_distribution[cls] = count
            total_samples += count
            
            # Quick profile of first few images
            for img_path in images[:5]:
                try:
                    with Image.open(img_path) as img:
                        sampled_widths.append(img.width)
                        sampled_heights.append(img.height)
                except Exception:
                    corrupted += 1
                    
        # Calculate stats
        avg_width = sum(sampled_widths) // len(sampled_widths) if sampled_widths else 224
        avg_height = sum(sampled_heights) // len(sampled_heights) if sampled_heights else 224
        
        # Check imbalance (max class size > 2x min class size)
        counts = list(class_distribution.values())
        imbalanced = False
        if counts and max(counts) > (min(counts) * 2):
            imbalanced = True
            
        return DatasetProfile(
            dataset_type="image",
            num_samples=total_samples,
            num_classes=num_classes,
            class_distribution=class_distribution,
            imbalanced=imbalanced,
            avg_width=avg_width,
            avg_height=avg_height,
            corrupted_images=corrupted
        )
