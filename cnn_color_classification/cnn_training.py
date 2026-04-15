import os
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

DATASET_DIR = "training_data"
IMG_SIZE = 64

X = []
y = []

classes = sorted(os.listdir(DATASET_DIR))
print("Training Classes:", classes)

for label, cls in enumerate(classes):
    folder = os.path.join(DATASET_DIR, cls)

    for file in os.listdir(folder):
        if file.lower().endswith(("jpg", "jpeg", "png")):
            img_path = os.path.join(folder, file)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            X.append(img)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Train Dataset Loaded:", X.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_train = to_categorical(y_train, num_classes=len(classes))
y_val = to_categorical(y_val, num_classes=len(classes))

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val),
    shuffle=True
)

# Show final training accuracy
final_train_acc = history.history["accuracy"][-1] * 100
print(f"\nFinal Training Accuracy: {final_train_acc:.2f}%")

model.save("bb_cnn_colors.h5")
joblib.dump(classes, "bb_cnn_colors_classes.pkl")

print("\nSaved bb_cnn_colors.h5")
print("Saved bb_cnn_colors_classes.pkl")
