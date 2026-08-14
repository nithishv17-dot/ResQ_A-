"""
inference.py - Production Inference Pipeline for Skin Lesion / Bug Bite Images

This script:
1. Preprocesses local skin images (resizing to 224x224, applying training RGB channel mean subtraction).
2. Obtains prediction probability vectors from each of the 4 base models.
3. Concatenates the probability outputs into a classifier fusion vector.
4. Evaluates the Stacking Ensemble Meta-Learner to output final bug bite prediction & confidence score.
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf

from preprocessing import apply_channel_mean_subtraction, IMG_SIZE
from models import (
    build_mobilenet_v2,
    build_vgg16,
    build_inception_v3_mixed4,
    build_densenet169,
    build_stacking_ensemble
)

class BugBiteClassifier:
    def __init__(self, processed_dir="./processed"):
        self.processed_dir = processed_dir
        self.models_dir = os.path.join(processed_dir, "models")
        self.class_map_path = os.path.join(processed_dir, "class_names.json")
        self.means_path = os.path.join(processed_dir, "channel_means.json")

        self.class_names = self._load_json(self.class_map_path, default=[
            "ants", "bed_bugs", "chiggers", "fleas", "mosquitos", "no_bites", "spiders", "ticks"
        ])
        self.channel_means = self._load_json(self.means_path, default={
            "R_mean": 123.68, "G_mean": 116.78, "B_mean": 103.94
        })

        self.num_classes = len(self.class_names)
        print(f"Loaded {self.num_classes} classes: {self.class_names}")

        self._load_models()

    def _load_json(self, path, default):
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        print(f"Notice: Config file not found at {path}. Using default fallback values.")
        return default

    def _load_models(self):
        print("\nLoading Deep Learning Models & Weights...")
        self.base_models = {}

        model_specs = [
            ("MobileNet-v2", build_mobilenet_v2, "mobilenetv2_bugbite.keras"),
            ("VGG16", build_vgg16, "vgg16_bugbite.keras"),
            ("Inception-v3 (Mixed4)", build_inception_v3_mixed4, "inceptionv3_mixed4_bugbite.keras"),
            ("DenseNet169", build_densenet169, "densenet169_bugbite.keras"),
        ]

        for name, build_fn, filename in model_specs:
            weight_path = os.path.join(self.models_dir, filename)
            model = build_fn(num_classes=self.num_classes)

            if os.path.exists(weight_path):
                model.load_weights(weight_path)
                print(f"  [OK] Loaded trained weights for {name} from {filename}")
            else:
                print(f"  [!] Warning: Weights file not found for {name} ({weight_path}). Using uninitialized weights.")
            
            self.base_models[name] = model

        # Load Stacking Ensemble
        stack_weight_path = os.path.join(self.models_dir, "stacking_ensemble_bugbite.keras")
        self.stacking_model = build_stacking_ensemble(num_classes=self.num_classes, num_models=4)
        if os.path.exists(stack_weight_path):
            self.stacking_model.load_weights(stack_weight_path)
            print(f"  [OK] Loaded Stacking Ensemble Meta-Learner from {stack_weight_path}")
        else:
            print(f"  [!] Warning: Stacking Ensemble weights file not found ({stack_weight_path}). Using uninitialized weights.")

    def preprocess_image(self, image_path):
        """
        Loads and preprocesses single image tensor with RGB mean subtraction.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)  # (224, 224, 3)
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        # Apply channel mean subtraction
        norm_tensor = apply_channel_mean_subtraction(img_tensor, self.channel_means)
        batch_tensor = tf.expand_dims(norm_tensor, axis=0)  # (1, 224, 224, 3)
        return batch_tensor

    def predict(self, image_path):
        """
        Runs full pipeline inference on single skin lesion image.
        """
        input_tensor = self.preprocess_image(image_path)
        base_probs_list = []
        individual_results = {}

        # 1. Base model predictions
        for name, model in self.base_models.items():
            prob = model.predict(input_tensor, verbose=0)  # Shape (1, num_classes)
            base_probs_list.append(prob)
            
            pred_idx = int(np.argmax(prob[0]))
            confidence = float(prob[0][pred_idx]) * 100
            individual_results[name] = {
                "predicted_class": self.class_names[pred_idx],
                "confidence": confidence,
                "probabilities": {cls: float(p) for cls, p in zip(self.class_names, prob[0])}
            }

        # 2. Concatenate probabilities into fusion vector of shape (1, 4 * num_classes)
        stacked_vector = np.hstack(base_probs_list)

        # 3. Stacking Ensemble prediction
        ensemble_prob = self.stacking_model.predict(stacked_vector, verbose=0)[0]
        ensemble_pred_idx = int(np.argmax(ensemble_prob))
        ensemble_confidence = float(ensemble_prob[ensemble_pred_idx]) * 100

        result = {
            "image_path": image_path,
            "predicted_class": self.class_names[ensemble_pred_idx],
            "confidence": ensemble_confidence,
            "ensemble_probabilities": {cls: float(p) for cls, p in zip(self.class_names, ensemble_prob)},
            "base_models_breakdown": individual_results
        }

        return result

def print_result_summary(result):
    print("\n" + "=" * 60)
    print(f"BUG BITE PREDICTION REPORT")
    print("=" * 60)
    print(f"Image:            {result['image_path']}")
    print(f"FINAL PREDICTION: {result['predicted_class'].upper()}")
    print(f"CONFIDENCE SCORE: {result['confidence']:.2f}%")
    print("-" * 60)
    print("Individual Model Predictions:")
    for model_name, info in result["base_models_breakdown"].items():
        print(f"  - {model_name:<24}: {info['predicted_class']:<15} ({info['confidence']:.2f}%)")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference tool for Bug Bite Deep Learning Classifier")
    parser.add_argument("--image", type=str, required=True, help="Path to input skin image")
    parser.add_argument("--processed_dir", type=str, default="./processed", help="Path to directory containing models and metadata")
    
    args = parser.parse_args()

    classifier = BugBiteClassifier(processed_dir=args.processed_dir)
    res = classifier.predict(args.image)
    print_result_summary(res)
