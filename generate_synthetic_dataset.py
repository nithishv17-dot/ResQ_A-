"""
generate_synthetic_dataset.py - Utility to generate synthetic dataset for verification testing

Generates a lightweight dataset structure with synthetic RGB images for each bug bite class.
Can write to local `./sample_dataset` or to `D:\\datasets\\datasets\\external\\stage1\\insect-bite-dataset`.
"""

import os
import argparse
import numpy as np
from PIL import Image

DEFAULT_CLASSES = [
    "ants", "bed_bugs", "chiggers", "fleas",
    "mosquitos", "no_bites", "spiders", "ticks"
]

def generate_dataset(target_dir="./sample_dataset", num_train_per_class=4, num_test_per_class=2):
    train_dir = os.path.join(target_dir, "training")
    test_dir = os.path.join(target_dir, "testing")

    print(f"Generating synthetic dataset in: {target_dir}")

    for cls in DEFAULT_CLASSES:
        cls_train_path = os.path.join(train_dir, cls)
        cls_test_path = os.path.join(test_dir, cls)
        os.makedirs(cls_train_path, exist_ok=True)
        os.makedirs(cls_test_path, exist_ok=True)

        # Seed per class for distinct colors
        cls_seed = hash(cls) % 255

        # Create train images
        for i in range(num_train_per_class):
            img_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint32)
            img_data[:, :, 0] = (img_data[:, :, 0] + cls_seed) % 256  # Color tint
            img = Image.fromarray(img_data.astype(np.uint8))
            img.save(os.path.join(cls_train_path, f"sample_train_{i+1}.jpg"))

        # Create test images
        for i in range(num_test_per_class):
            img_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint32)
            img_data[:, :, 1] = (img_data[:, :, 1] + cls_seed) % 256
            img = Image.fromarray(img_data.astype(np.uint8))
            img.save(os.path.join(cls_test_path, f"sample_test_{i+1}.jpg"))

    print(f"Synthetic dataset successfully created at {target_dir}")
    print(f"  - Classes ({len(DEFAULT_CLASSES)}): {DEFAULT_CLASSES}")
    print(f"  - Training samples per class: {num_train_per_class}")
    print(f"  - Testing samples per class: {num_test_per_class}")
    return train_dir, test_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for bug bite classification testing")
    parser.add_argument("--output_dir", type=str, default="./sample_dataset", help="Target directory for synthetic dataset")
    parser.add_argument("--train_count", type=int, default=4, help="Number of training images per class")
    parser.add_argument("--test_count", type=int, default=2, help="Number of test images per class")

    args = parser.parse_args()
    generate_dataset(args.output_dir, args.train_count, args.test_count)
