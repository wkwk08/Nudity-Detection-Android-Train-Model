import os, cv2, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models

# Load HSV pixels from tone-grouped skin samples
def load_hsv_training_data(base_path="datasets/ducnguyen168/data_skintone", sample_size=10000):
    tone_groups = ["dark", "mid-dark", "mid-light", "light"]
    hsv_pixels, labels = [], []
    samples_per_group = sample_size // len(tone_groups)

    for tone in tone_groups:
        folder = os.path.join(base_path, tone)
        files = os.listdir(folder)[:samples_per_group]
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None: continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            pixels = hsv.reshape(-1, 3)
            hsv_pixels.append(pixels)
            labels.append(np.ones(len(pixels)))  # Assume all pixels are skin

    X = np.vstack(hsv_pixels) / 255.0  # Normalize HSV
    y = np.hstack(labels)
    return X, y

# Build a simple MLP classifier
def build_hsv_classifier():
    model = models.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and export
X, y = load_hsv_training_data()
model = build_hsv_classifier()
model.fit(X, y, epochs=5, batch_size=256)
model.save("hsv_skin_classifier.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("hsv_skin_classifier.tflite", "wb") as f:
    f.write(tflite_model)

print("Exported hsv_skin_classifier.tflite")
