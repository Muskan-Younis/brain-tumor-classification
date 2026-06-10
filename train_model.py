import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PATHS
# -----------------------------
train_path = "cleaned_dataset/Training"
test_path = "cleaned_dataset/Testing"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 35

# -----------------------------
# DATA AUGMENTATION
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,

    rotation_range=25,

    zoom_range=0.2,

    width_shift_range=0.2,
    height_shift_range=0.2,

    shear_range=0.15,

    horizontal_flip=True,

    brightness_range=[0.75, 1.25],

    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# -----------------------------
# LOAD DATA
# -----------------------------
train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb'
)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

# -----------------------------
# CLASS WEIGHTS
# -----------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))

print("\nClass Weights:")
print(class_weights)

# -----------------------------
# BASE MODEL
# -----------------------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# -----------------------------
# FINE TUNING
# -----------------------------
base_model.trainable = True

# Unfreeze more layers
for layer in base_model.layers[:-50]:
    layer.trainable = False

# -----------------------------
# MODEL
# -----------------------------
model = models.Sequential([

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.BatchNormalization(),

    layers.Dense(
        256,
        activation='relu'
    ),

    layers.Dropout(0.4),

    layers.Dense(
        train_data.num_classes,
        activation='softmax'
    )

])

# -----------------------------
# COMPILE
# -----------------------------
model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00005
    ),

    loss=tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.1
    ),

    metrics=[

        'accuracy',

        tf.keras.metrics.Recall(name='recall'),

        tf.keras.metrics.Precision(name='precision')

    ]
)

model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------
early_stop = EarlyStopping(

    monitor='val_loss',

    patience=7,

    restore_best_weights=True

)

checkpoint = ModelCheckpoint(

    "brain_tumor_model.h5",

    monitor='val_loss',

    save_best_only=True,

    verbose=1
)

reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.3,

    patience=3,

    min_lr=1e-7,

    verbose=1
)

# -----------------------------
# TRAIN
# -----------------------------
history = model.fit(

    train_data,

    validation_data=test_data,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]
)

# -----------------------------
# EVALUATION
# -----------------------------
loss, accuracy, recall, precision = model.evaluate(test_data)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"Precision : {precision:.4f}")

# -----------------------------
# PREDICTIONS
# -----------------------------
y_pred = model.predict(test_data)

y_pred_classes = np.argmax(y_pred, axis=1)

y_true = test_data.classes

class_labels = list(
    test_data.class_indices.keys()
)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(
    y_true,
    y_pred_classes
)

plt.figure(figsize=(8,6))

sns.heatmap(

    cm,

    annot=True,

    fmt='d',

    xticklabels=class_labels,

    yticklabels=class_labels
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
print("\nClassification Report:\n")

print(

    classification_report(

        y_true,

        y_pred_classes,

        target_names=class_labels
    )
)

print("\nBest model saved as brain_tumor_model.h5")