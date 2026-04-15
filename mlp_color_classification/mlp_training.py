import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib


TRAIN_DIR = "training_data"
IMG_SIZE = 64

X = []
y = []

classes = sorted(os.listdir(TRAIN_DIR))
print("Classes:", classes)


for label, cls in enumerate(classes):
    folder = os.path.join(TRAIN_DIR, cls)

    for file in os.listdir(folder):
        if file.lower().endswith(("jpg", "jpeg", "png")):
            path = os.path.join(folder, file)

            img = cv2.imread(path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img.flatten())
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Training samples:", len(X))


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


clf = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    verbose=True
)

print("Training model...")
clf.fit(X_scaled, y)


train_pred = clf.predict(X_scaled)
train_acc = accuracy_score(y, train_pred)

print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")


joblib.dump(clf, "mlp_hat_color.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(classes, "classes.pkl")

print("\nModel Saved Successfully.")