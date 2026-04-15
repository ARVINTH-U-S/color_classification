import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

TEST_DIR = "testing_images"
IMG_SIZE = 64

clf = joblib.load("mlp_hat_color.pkl")
scaler = joblib.load("scaler.pkl")
classes = joblib.load("classes.pkl")

X_test = []
y_test = []

print("Training Classes:", classes)

# -----------------------------
# LOAD TEST IMAGES (Skip missing folders)
# -----------------------------
for label, cls in enumerate(classes):
    folder = os.path.join(TEST_DIR, cls)

    if not os.path.exists(folder):
        print(f"[WARNING] Missing test folder: {cls} → Skipping.")
        continue

    for file in os.listdir(folder):
        if file.lower().endswith(("jpg", "jpeg", "png")):
            path = os.path.join(folder, file)

            img = cv2.imread(path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.flatten()

            X_test.append(img)
            y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

print("\nTotal Test Samples:", len(X_test))


X_test = scaler.transform(X_test)

pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"\nTest Accuracy: {acc * 100:.2f}%")

cm = confusion_matrix(y_test, pred)

present_labels = sorted(list(set(y_test)))

cm = cm[:, present_labels]        
cm = cm[present_labels, :]      

present_class_names = [classes[i] for i in present_labels]

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=present_class_names,
    yticklabels=present_class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("MLP Confusion Matrix")

plt.savefig("final_confusion_matrix.jpg", dpi=300)
plt.close()

print("\nConfusion matrix saved as: final_confusion_matrix.jpg")