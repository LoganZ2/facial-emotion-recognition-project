from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import cv2
import os
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def process_image(img_path):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (48, 48))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)
            return img
        else:
            print(f"Warning: Unable to read image {img_path}")
            return None
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

def load_and_split_data(data_dir):
    emotions = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sadness': 4, 'surprise': 5, 'contempt': 6}
    X_train, y_train = [], []
    X_public_test, y_public_test = [], []
    X_private_test, y_private_test = [], []

    for emotion, label in emotions.items():
        print(f"Processing emotion: {emotion}")
        emotion_dir = os.path.join(data_dir, emotion)
        img_paths = [os.path.join(emotion_dir, img_name) for img_name in os.listdir(emotion_dir)]

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_image, img_paths))

        results = [result for result in results if result is not None]

        X_temp, X_private, y_temp, y_private = train_test_split(results, [label] * len(results), test_size=0.1, random_state=42)
        X_train_split, X_public, y_train_split, y_public = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42)  # 0.1111 ≈ 10/90

        X_train.extend(X_train_split)
        y_train.extend(y_train_split)
        X_public_test.extend(X_public)
        y_public_test.extend(y_public)
        X_private_test.extend(X_private)
        y_private_test.extend(y_private)

    return (np.array(X_train), np.array(y_train),
            np.array(X_public_test), np.array(y_public_test),
            np.array(X_private_test), np.array(y_private_test))

data_dir = '/content/drive/MyDrive/Colab Notebooks/archive1/'

X_train, y_train, X_public_test, y_public_test, X_private_test, y_private_test = load_and_split_data(data_dir)

y_train = to_categorical(y_train, num_classes=7)
y_public_test = to_categorical(y_public_test, num_classes=7)
y_private_test = to_categorical(y_private_test, num_classes=7)

print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Public Test data shape: {X_public_test.shape}, Public Test labels shape: {y_public_test.shape}")
print(f"Private Test data shape: {X_private_test.shape}, Private Test labels shape: {y_private_test.shape}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

history = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                    validation_data=(X_public_test, y_public_test),
                    epochs=200,
                    verbose=2)

model.save('emotion_recognition_model.h5')

loss, accuracy = model.evaluate(X_private_test, y_private_test, verbose=0)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

y_pred = model.predict(X_private_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_private_test, axis=1)

conf_matrix = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'contempt'], yticklabels=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'contempt'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import classification_report
import numpy as np
y_pred = model.predict(X_private_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_private_test, axis=1)
report = classification_report(y_true, y_pred_classes, target_names=['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'contempt'])
print("Classification Report:\n", report)