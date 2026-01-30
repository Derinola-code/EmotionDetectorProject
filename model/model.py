import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# === 1. SMART PATHING (Fixes WinError 3) ===
# This finds the 'EmotionDectorProject' root even if you run from the wrong folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Goes up one level to Project Root
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "images")

train_dir = os.path.join(DATASET_PATH, "train")
val_dir = os.path.join(DATASET_PATH, "val")

# Safety Verification
print(f"Checking for images at: {train_dir}")
if not os.path.exists(train_dir):
    print(f"❌ ERROR: Cannot find folder: {train_dir}")
    print("Ensure 'dataset' is in your main 'EmotionDectorProject' folder.")
    exit()

# === 2. DATA PREPARATION ===
IMG_SIZE = (48, 48)
BATCH_SIZE = 32

datagen = ImageDataGenerator(rescale=1./255)

# Standard Keras directory flow
train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# === 3. CNN MODEL ARCHITECTURE ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax') # 7 emotions in FER2013
])

model.compile(optimizer=Adam(learning_rate=0.0005), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# === 4. SAVE & TRAIN ===
MODEL_NAME = os.path.join(SCRIPT_DIR, "face_emotionModel.h5")
checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_accuracy', save_best_only=True, verbose=1)

print("🚀 Starting Training...")
model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=[checkpoint])
print(f"✅ Success! Model saved as: {MODEL_NAME}")