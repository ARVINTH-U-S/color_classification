import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import joblib

DATASET_DIR = "training_data"
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 20

train_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

num_classes = len(train_gen.class_indices)
print("Classes:", train_gen.class_indices)

base = EfficientNetV2L(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.4)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base.input, outputs=output)

model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    train_gen,
    steps_per_epoch=100,
    epochs=EPOCHS
)

model.save("efficientnet_color_model.h5")
joblib.dump(train_gen.class_indices, "efficientnet_class_indices.pkl")

print("Training complete!")
