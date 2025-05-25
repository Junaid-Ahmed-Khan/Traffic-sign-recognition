import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 64
DATASET_DIR = "gtsrb_subset"

CLASS_NAMES = sorted(os.listdir(DATASET_DIR))
NUM_CLASSES = len(CLASS_NAMES)
print("🔍 Found classes:", CLASS_NAMES)

# Save class names
with open("class_names.txt", "w") as f:
    for name in CLASS_NAMES:
        f.write(name + "\n")

images, labels = [], []
for idx, class_name in enumerate(CLASS_NAMES):
    class_path = os.path.join(DATASET_DIR, class_name)
    for fname in os.listdir(class_path):
        if fname.lower().endswith((".ppm", ".png", ".jpg", ".jpeg")):
            try:
                img = Image.open(os.path.join(class_path, fname)).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE))
                images.append(np.array(img) / 255.0)
                labels.append(idx)
            except Exception as e:
                print(f"⚠️ Skipping {fname}: {e}")

images = np.array(images)
labels = to_categorical(labels, NUM_CLASSES)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
np.savez("gtsrb_subset.npz", x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
print("✅ Dataset saved to gtsrb_subset.npz")
