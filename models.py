"""
models.py - Transfer Learning Base Networks & Stacking Ensemble Architecture

This module defines:
1. MobileNet-v2 (Lightweight, 88 layers) with custom Conv2D & Dense top head.
2. VGG16 (First 10 layers frozen, retrained from layer 10+).
3. Inception-v3 Mixed4 Variant (Sliced at 'mixed4' layer, 110,592 bottleneck features).
4. DenseNet169 (169 layers, fully frozen base, BatchNorm + multi-stage Dense top head).
5. Stacking Ensemble Meta-Learner (Classifier fusion head concatenating base probability vectors).
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2, VGG16, InceptionV3, DenseNet169

def build_mobilenet_v2(num_classes, input_shape=(224, 224, 3), learning_rate=3e-7):
    """
    Model 1: MobileNet-v2
    - Base model layers fully frozen.
    - Top head: Conv2D (256, 3x3) -> GlobalAvgPool -> Dense(64, 10% Dropout) -> Dense(32) -> Softmax
    - Optimizer: Adam(lr=3e-7)
    """
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Fully frozen

    x = base_model.output  # 7x7x1280 feature maps
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="mnet_conv2d_256")(x)
    x = layers.GlobalAveragePooling2D(name="mnet_gap")(x)  # 256 vector
    x = layers.Dense(64, activation="relu", name="mnet_dense_64")(x)
    x = layers.Dropout(0.10, name="mnet_dropout_10")(x)
    x = layers.Dense(32, activation="relu", name="mnet_dense_32")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="mnet_predictions")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name="MobileNetV2_BugBite")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_vgg16(num_classes, input_shape=(224, 224, 3), learning_rate=1e-5):
    """
    Model 2: VGG16
    - Freeze only first 10 layers; unfreeze and retrain remaining base layers (10+).
    - Flatten output of final MaxPooling2D layer.
    - Top head: Dense(512, 20% Dropout) -> Dense(256) -> Softmax
    - Optimizer: Adam(lr=1e-5)
    """
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze first 10 layers, unfreeze remaining
    for layer in base_model.layers[:10]:
        layer.trainable = False
    for layer in base_model.layers[10:]:
        layer.trainable = True

    x = base_model.output
    x = layers.Flatten(name="vgg_flatten")(x)
    x = layers.Dense(512, activation="relu", name="vgg_dense_512")(x)
    x = layers.Dropout(0.20, name="vgg_dropout_20")(x)
    x = layers.Dense(256, activation="relu", name="vgg_dense_256")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="vgg_predictions")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name="VGG16_BugBite")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_inception_v3_mixed4(num_classes, input_shape=(224, 224, 3), learning_rate=1e-4):
    """
    Model 3: Inception-v3 (Mixed 4 Variant)
    - Truncated at layer 'mixed4' (output shape 12x12x768 = 110,592 features).
    - Freeze all layers from input up to 'mixed4'.
    - Top head: Flatten -> Dense(1024) -> Dropout(20%) -> Softmax
    - Optimizer: Adam(lr=1e-4)
    """
    base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=input_shape)
    
    # Get output of 'mixed4' bottleneck layer
    mixed4_layer = base_model.get_layer("mixed4")
    truncated_model = Model(inputs=base_model.input, outputs=mixed4_layer.output, name="InceptionV3_Mixed4_Base")

    # Freeze all layers up to and including 'mixed4'
    for layer in truncated_model.layers:
        layer.trainable = False

    x = truncated_model.output
    x = layers.Flatten(name="incept_flatten")(x)  # 110,592 dimensions
    x = layers.Dense(1024, activation="relu", name="incept_dense_1024")(x)
    x = layers.Dropout(0.20, name="incept_dropout_20")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="incept_predictions")(x)

    model = Model(inputs=truncated_model.input, outputs=outputs, name="InceptionV3_Mixed4_BugBite")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_densenet169(num_classes, input_shape=(224, 224, 3), learning_rate=1e-5):
    """
    Model 4: DenseNet169
    - Base model layers fully frozen.
    - Top head: GlobalAvgPool -> BatchNorm -> Dropout(20%) -> Dense(2048, 20% Dropout) -> Dense(512, 20% Dropout) -> Dense(128) -> Softmax
    - Optimizer: Adam(lr=1e-5)
    """
    base_model = DenseNet169(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Fully frozen

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="dnet_gap")(x)
    x = layers.BatchNormalization(name="dnet_bn")(x)
    x = layers.Dropout(0.20, name="dnet_dropout_1")(x)
    x = layers.Dense(2048, activation="relu", name="dnet_dense_2048")(x)
    x = layers.Dropout(0.20, name="dnet_dropout_2")(x)
    x = layers.Dense(512, activation="relu", name="dnet_dense_512")(x)
    x = layers.Dropout(0.20, name="dnet_dropout_3")(x)
    x = layers.Dense(128, activation="relu", name="dnet_dense_128")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dnet_predictions")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name="DenseNet169_BugBite")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_stacking_ensemble(num_classes, num_models=4, learning_rate=1e-3):
    """
    Classifier Fusion Stacking Ensemble Head
    - Input: Combined probability vector of shape (num_models * num_classes)
    - Dense(64, ReLU) -> Dense(num_classes, Softmax)
    - Loss: Categorical Cross-Entropy
    """
    input_dim = num_models * num_classes
    inputs = layers.Input(shape=(input_dim,), name="stacked_prob_features")
    x = layers.Dense(64, activation="relu", name="stack_dense_64")(inputs)
    outputs = layers.Dense(num_classes, activation="softmax", name="stack_predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="StackingEnsemble_BugBite")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
