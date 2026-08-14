"""
train.py - Pipeline Training, Data Augmentation, Hyperparameter Setups, & Model Checkpoints

This script orchestrates:
1. RGB Channel Mean calculation & metadata saving.
2. Training of the 4 base models with paper-specified hyperparameters & callbacks.
3. Probability vector extraction from base models.
4. Training and evaluation of the Stacking Ensemble Meta-Learner.
5. Saving all trained model artifacts and performance reports.
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf

from preprocessing import (
    DEFAULT_TRAIN_DIR,
    DEFAULT_TEST_DIR,
    get_class_names,
    calculate_channel_means,
    load_dataset
)
from models import (
    build_mobilenet_v2,
    build_vgg16,
    build_inception_v3_mixed4,
    build_densenet169,
    build_stacking_ensemble
)

def extract_probabilities_and_labels(model, dataset):
    """
    Extracts prediction probability vectors from a model over a dataset,
    along with true class label vectors.
    """
    probs_list = []
    labels_list = []

    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        probs_list.append(preds)
        labels_list.append(y_batch.numpy())

    probs = np.concatenate(probs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return probs, labels

def run_training_pipeline(args):
    train_dir = args.train_dir
    test_dir = args.test_dir
    output_dir = args.output_dir
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 70)
    print("BUG BITE CLASSIFICATION: END-TO-END DEEP LEARNING PIPELINE")
    print("=" * 70)

    # 1. Dynamic Class Detection
    class_names = get_class_names(train_dir)
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    # Save class names mapping
    class_map_path = os.path.join(output_dir, "class_names.json")
    with open(class_map_path, "w") as f:
        json.dump(class_names, f, indent=4)
    print(f"Saved class names to {class_map_path}")

    # 2. RGB Channel Means Calculation & Subtraction setup
    means_json_path = os.path.join(output_dir, "channel_means.json")
    if os.path.exists(means_json_path) and not args.recalculate_means:
        with open(means_json_path, "r") as f:
            channel_means = json.load(f)
        print(f"Loaded existing channel means from {means_json_path}: {channel_means}")
    else:
        channel_means = calculate_channel_means(train_dir, save_path=means_json_path)

    # 3. Load Datasets
    print("\n--- Loading Training & Testing Datasets ---")
    train_ds, _ = load_dataset(
        train_dir,
        batch_size=args.batch_size,
        is_training=True,
        channel_means=channel_means,
        augment=True
    )

    # Load non-augmented version of train set for stacked feature extraction
    train_ds_no_aug, _ = load_dataset(
        train_dir,
        batch_size=args.batch_size,
        is_training=False,
        channel_means=channel_means,
        augment=False
    )

    test_ds, _ = load_dataset(
        test_dir,
        batch_size=args.batch_size,
        is_training=False,
        channel_means=channel_means,
        augment=False
    )

    epochs_mnet = 1 if args.dry_run else args.epochs_mobilenet
    epochs_vgg = 1 if args.dry_run else args.epochs_vgg
    epochs_incept = 1 if args.dry_run else args.epochs_inception
    epochs_dnet = 1 if args.dry_run else args.epochs_densenet
    epochs_stack = 1 if args.dry_run else args.epochs_stacking

    train_prob_vectors = []
    test_prob_vectors = []
    y_train_stack = None
    y_test_stack = None

    # Helper function to train base models
    def train_base_model(name, build_fn, epochs, filename):
        nonlocal y_train_stack, y_test_stack
        print(f"\n" + "-" * 50)
        print(f"Training Model: {name} (Epochs: {epochs})")
        print("-" * 50)

        model = build_fn(num_classes=num_classes)
        save_path = os.path.join(models_dir, filename)

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        )

        history = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=epochs,
            callbacks=[checkpoint_cb]
        )

        # Load best weights
        if os.path.exists(save_path):
            model.load_weights(save_path)

        # Evaluate best model
        loss, acc = model.evaluate(test_ds, verbose=0)
        print(f"[{name}] Best Validation Accuracy: {acc * 100:.2f}%, Loss: {loss:.4f}")

        # Extract probabilities for Stacking Meta-Learner
        print(f"[{name}] Extracting prediction probabilities for Stacking Ensemble...")
        train_probs, y_tr = extract_probabilities_and_labels(model, train_ds_no_aug)
        test_probs, y_te = extract_probabilities_and_labels(model, test_ds)

        if y_train_stack is None:
            y_train_stack = y_tr
            y_test_stack = y_te

        train_prob_vectors.append(train_probs)
        test_prob_vectors.append(test_probs)

        return model

    # 4. Train 4 Base Models
    print("\n==================================================")
    print("STAGE 1: BASE MODEL TRANSFER LEARNING & FINE-TUNING")
    print("==================================================")

    train_base_model("MobileNet-v2", build_mobilenet_v2, epochs_mnet, "mobilenetv2_bugbite.keras")
    train_base_model("VGG16", build_vgg16, epochs_vgg, "vgg16_bugbite.keras")
    train_base_model("Inception-v3 (Mixed4)", build_inception_v3_mixed4, epochs_incept, "inceptionv3_mixed4_bugbite.keras")
    train_base_model("DenseNet169", build_densenet169, epochs_dnet, "densenet169_bugbite.keras")

    # 5. Stacking Ensemble Concatenation & Training
    print("\n==================================================")
    print("STAGE 2: CLASSIFIER FUSION VIA STACKING ENSEMBLE")
    print("==================================================")

    # Concatenate horizontal probability matrices: (N, 4 * num_classes)
    X_train_stack = np.hstack(train_prob_vectors)
    X_test_stack = np.hstack(test_prob_vectors)

    print(f"Stacking Feature Shape (Train): {X_train_stack.shape}")
    print(f"Stacking Feature Shape (Test):  {X_test_stack.shape}")

    stacking_model = build_stacking_ensemble(num_classes=num_classes, num_models=4)
    stack_save_path = os.path.join(models_dir, "stacking_ensemble_bugbite.keras")

    stack_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=stack_save_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    stack_history = stacking_model.fit(
        X_train_stack,
        y_train_stack,
        validation_data=(X_test_stack, y_test_stack),
        epochs=epochs_stack,
        batch_size=args.batch_size,
        callbacks=[stack_checkpoint_cb]
    )

    if os.path.exists(stack_save_path):
        stacking_model.load_weights(stack_save_path)

    final_loss, final_acc = stacking_model.evaluate(X_test_stack, y_test_stack, verbose=0)

    print("\n==================================================")
    print("FINAL TRAINING RESULTS & EVALUATION SUMMARY")
    print("==================================================")
    print(f"Stacking Ensemble Classifier Fusion Accuracy: {final_acc * 100:.2f}%")
    print(f"All models saved under directory: {models_dir}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bug Bite Classification End-to-End Deep Learning System")
    parser.add_argument("--train_dir", type=str, default=DEFAULT_TRAIN_DIR, help="Path to training set directory")
    parser.add_argument("--test_dir", type=str, default=DEFAULT_TEST_DIR, help="Path to testing set directory")
    parser.add_argument("--output_dir", type=str, default="./processed", help="Path to save artifacts & models")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs_mobilenet", type=int, default=150, help="Epochs for MobileNet-v2")
    parser.add_argument("--epochs_vgg", type=int, default=50, help="Epochs for VGG16")
    parser.add_argument("--epochs_inception", type=int, default=50, help="Epochs for Inception-v3 Mixed4")
    parser.add_argument("--epochs_densenet", type=int, default=70, help="Epochs for DenseNet169")
    parser.add_argument("--epochs_stacking", type=int, default=30, help="Epochs for Stacking Ensemble")
    parser.add_argument("--recalculate_means", action="store_true", help="Force recalculation of RGB channel means")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 epoch per model for rapid testing")

    args = parser.parse_args()
    run_training_pipeline(args)
