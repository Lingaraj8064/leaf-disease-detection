import os
import cv2
import numpy as np

def load_data(data_dir, img_size=(128, 128)):
    data = []
    labels = []
    classes = os.listdir(data_dir)

    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)  # Read image
            if img is None:  # Skip corrupt images
                continue
            img = cv2.resize(img, img_size)  # Resize to (128,128)
            img = img.astype("float32") / 255.0  # Normalize pixel values
            data.append(img)
            labels.append(class_idx)

    data = np.array(data)  # Convert list to NumPy array
    labels = np.array(labels)

    return data, labels

# Example usage
X_train, y_train = load_data('../dataset/train/')
X_test, y_test = load_data('../dataset/test/')

print("Training data shape:", X_train.shape)  # Should be (num_samples, 128, 128, 3)
print("Testing data shape:", X_test.shape)  # Should be (num_samples, 128, 128, 3)
