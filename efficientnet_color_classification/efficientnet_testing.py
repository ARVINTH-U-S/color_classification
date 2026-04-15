import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import load_model

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


TEST_DIR = "Dtesting_images"
MODEL_PATH = "efficientnet_color_model.h5"
CONF_MATRIX_NAME = "efficientnet_confusion_matrix.png"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# These are your **training classes (7 total)**
training_classes = ["gray", "white", "red", "yellow", "blue", "brown", "green"]

print("Loading model...")
model = load_model(MODEL_PATH)

print("Loading test dataset...")
test_ds = image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='int',
    shuffle=False,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_class_names = test_ds.class_names
print("\nTest dataset classes:", test_class_names)

label_map = {name: training_classes.index(name) for name in test_class_names}

inv_label_map = {training_classes.index(name): name for name in test_class_names}


y_true = []
for _, labels in test_ds:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)

y_true_mapped = np.array([label_map[test_class_names[i]] for i in y_true])


print("\nPredicting...")
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)


valid_training_indices = [training_classes.index(c) for c in test_class_names]

y_pred_filtered = np.where(
    np.isin(y_pred, valid_training_indices),
    y_pred,
    -1 
)

mask = y_pred_filtered != -1
y_true_final = y_true_mapped[mask]
y_pred_final = y_pred_filtered[mask]


accuracy = accuracy_score(y_true_final, y_pred_final)
print("\n----------------------------------------------")
print(f"🔥 TEST ACCURACY: {accuracy * 100:.2f}%")
print("----------------------------------------------\n")

print("Classification Report (Only Test Classes):")
print(classification_report(
    y_true_final,
    y_pred_final,
    labels=valid_training_indices,
    target_names=test_class_names,
    zero_division=0
))


cm = confusion_matrix(
    y_true_final,
    y_pred_final,
    labels=valid_training_indices
)

plt.figure(figsize=(9, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=test_class_names,
    yticklabels=test_class_names
)

plt.title("EfficientNet Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.savefig(CONF_MATRIX_NAME, dpi=300)
plt.show()

print(f"\nSaved confusion matrix as {CONF_MATRIX_NAME}")
