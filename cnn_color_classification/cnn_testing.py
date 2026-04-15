import os
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

TEST_DIR = r"D:\tatva\dev_test\testing_images"
IMG_SIZE = 64

# Load trained model and class names
model = load_model("bb_cnn_colors.h5")
train_classes = joblib.load("bb_cnn_colors_classes.pkl")
print("Training Classes:", train_classes)

# Load only test dataset classes that exist
test_classes = sorted(os.listdir(TEST_DIR))
print("Test Classes:", test_classes)

X_test = []
y_test = []

# Map test class labels to training class indices
test_class_to_label = {cls: train_classes.index(cls) for cls in test_classes}

for cls in test_classes:
    folder = os.path.join(TEST_DIR, cls)

    for file in os.listdir(folder):
        if file.lower().endswith(("jpg", "jpeg", "png")):
            img_path = os.path.join(folder, file)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            X_test.append(img)
            y_test.append(test_class_to_label[cls])

X_test = np.array(X_test)
y_test = np.array(y_test)

print("Test Samples Loaded:", X_test.shape[0])

# ---- PREDICTION ----
pred = model.predict(X_test)
pred_labels = np.argmax(pred, axis=1)

# ---- ACCURACY ----
accuracy = (pred_labels == y_test).mean() * 100
print(f"\nTest Accuracy: {accuracy:.2f}%")

# ---- CONFUSION MATRIX FIX ----

# Only include the classes present in test dataset
label_ids = list(test_class_to_label.values())

cm = confusion_matrix(y_test, pred_labels, labels=label_ids)

# Display only test classes
display_labels = test_classes

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

plt.figure(figsize=(8, 6))
disp.plot(cmap="Blues", values_format="d")
plt.title("CNN confusion Matrix")
plt.savefig("final_cropped_bb_confusion_matrix.jpg")
plt.close()

print("Saved cropped_bb_confusion_matrix.jpg")




