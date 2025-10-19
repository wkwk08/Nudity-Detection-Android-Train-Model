import os, cv2, numpy as np, pandas as pd

def load_pratheepan_data(base_path="datasets/cs-chan", size=(256, 256)):
    image_dir = os.path.join(base_path, "images")
    mask_dir = os.path.join(base_path, "masks")
    images, masks = [], []
    for fname in os.listdir(image_dir):
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        if os.path.exists(mask_path):
            img = cv2.resize(cv2.imread(img_path), size)
            mask = cv2.resize(cv2.imread(mask_path, 0), size)
            images.append(img / 255.0)
            masks.append((mask > 0).astype(np.float32).reshape(size + (1,)))
    return np.array(images), np.array(masks)

def load_ducnguyen_data(base_path="datasets/ducnguyen168", size=(256, 256), sample_size=500):
    images = []
    files = os.listdir(base_path)[:sample_size]
    for fname in files:
        img_path = os.path.join(base_path, fname)
        img = cv2.resize(cv2.imread(img_path), size)
        images.append(img / 255.0)
    return np.array(images)

def load_fitzpatrick_data(base_path="datasets/fitzpatrick17k", size=(256, 256), sample_size=1000):
    df = pd.read_csv(os.path.join(base_path, "metadata.csv")).sample(n=sample_size, random_state=42)
    images, tones = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(base_path, "images", row["file_name"])
        if os.path.exists(img_path):
            img = cv2.resize(cv2.imread(img_path), size)
            images.append(img / 255.0)
            tones.append(row["fitzpatrick_type"])
    return np.array(images), tones
