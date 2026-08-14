"""
preprocessing.py - Preprocessing & Augmentation Pipeline for Bug Bite Classification

This module handles:
1. Dynamic class directory scanning.
2. Training RGB channel mean calculation & subtraction for skin tone / lighting normalization.
3. Data augmentation (Rotation [-10, 10] deg, Zoom 20%, Translation 10%, Horizontal & Vertical Flip).
4. Building optimized tf.data.Dataset pipelines for training and evaluation.
"""

import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

# Default paths as specified in research protocol & dataset specifications
DEFAULT_TRAIN_DIR = r"D:\datasets\datasets\external\stage1\insect-bite-dataset\training"
DEFAULT_TEST_DIR = r"D:\datasets\datasets\external\stage1\insect-bite-dataset\testing"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def get_class_names(dataset_dir):
    """
    Dynamically scans subdirectories in the dataset folder to get sorted class names.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    classes = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')
    ]
    classes.sort()
    return classes

def calculate_channel_means(train_dir, save_path=None):
    """
    Calculates the exact mean for each color channel (Red, Green, Blue) across
    all training dataset images to normalize lighting and skin tone variations.
    """
    print(f"Calculating RGB channel means from training set: {train_dir} ...")
    r_sum, g_sum, b_sum = 0.0, 0.0, 0.0
    total_pixels = 0

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = []

    for root, _, files in os.walk(train_dir):
        for f in files:
            if Path(f).suffix.lower() in image_extensions:
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        raise ValueError(f"No valid images found in training directory: {train_dir}")

    for img_path in image_paths:
        try:
            img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
            img_arr = tf.keras.utils.img_to_array(img)  # Shape (224, 224, 3), range [0, 255]
            r_sum += np.sum(img_arr[:, :, 0])
            g_sum += np.sum(img_arr[:, :, 1])
            b_sum += np.sum(img_arr[:, :, 2])
            total_pixels += IMG_SIZE[0] * IMG_SIZE[1]
        except Exception as e:
            print(f"Warning: Could not process image {img_path}: {e}")

    r_mean = float(r_sum / total_pixels)
    g_mean = float(g_sum / total_pixels)
    b_mean = float(b_sum / total_pixels)

    means = {"R_mean": r_mean, "G_mean": g_mean, "B_mean": b_mean}
    print(f"Calculated Training Channel Means (RGB): {means}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(means, f, indent=4)
        print(f"Saved channel means to {save_path}")

    return means

def load_channel_means(json_path):
    """
    Loads pre-calculated channel means from a JSON file.
    """
    with open(json_path, "r") as f:
        return json.load(f)

def create_augmentation_pipeline():
    """
    Creates data augmentation pipeline matching paper specifications:
    - Random rotation within [-10, 10] degrees (±10/360 ≈ 0.0278 factor)
    - Random zoom within range 0.2 (20%)
    - Width and height shifts up to 10% (0.1)
    - Horizontal and vertical flipping
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(factor=(-10/360, 10/360), fill_mode="nearest"),
        tf.keras.layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode="nearest"),
        tf.keras.layers.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode="nearest"),
        tf.keras.layers.RandomFlip("horizontal_and_vertical")
    ], name="data_augmentation")

def apply_channel_mean_subtraction(image, means):
    """
    Subtracts training RGB channel means from image tensor (0..255 float range).
    """
    r_mean = means.get("R_mean", 0.0)
    g_mean = means.get("G_mean", 0.0)
    b_mean = means.get("B_mean", 0.0)
    
    mean_tensor = tf.constant([r_mean, g_mean, b_mean], dtype=tf.float32)
    return tf.subtract(tf.cast(image, tf.float32), mean_tensor)

def load_dataset(dataset_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE, is_training=True, channel_means=None, augment=True):
    """
    Loads dataset from directory using image_dataset_from_directory, applies channel mean subtraction,
    one-hot encoding, and data augmentation for training sets.
    """
    class_names = get_class_names(dataset_dir)
    num_classes = len(class_names)

    ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        batch_size=batch_size,
        image_size=img_size,
        shuffle=is_training
    )

    augmentation_layer = create_augmentation_pipeline() if (is_training and augment) else None

    def preprocess_element(x, y):
        x = tf.cast(x, tf.float32)
        if channel_means is not None:
            r_m = channel_means.get("R_mean", 0.0)
            g_m = channel_means.get("G_mean", 0.0)
            b_m = channel_means.get("B_mean", 0.0)
            mean_tensor = tf.constant([r_m, g_m, b_m], dtype=tf.float32)
            x = x - mean_tensor
        
        if is_training and augmentation_layer is not None:
            x = augmentation_layer(x, training=True)
            
        return x, y

    ds = ds.map(preprocess_element, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return ds, class_names
