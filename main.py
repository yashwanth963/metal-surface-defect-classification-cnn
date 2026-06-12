import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras import layers, models

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

TRAIN_PATH = r"D:\project\dataset\train\images"
TEST_PATH  = r"D:\project\dataset\test\images"

# -----------------------------
# LOAD DATA (CLASS FOLDERS)
# -----------------------------
def load_class_folders(path):
    data = []
    labels = []

    print(f"\nReading from: {path}")
    classes = os.listdir(path)
    print("Classes found:", classes)

    for label_name in classes:
        class_path = os.path.join(path, label_name)

        if not os.path.isdir(class_path):
            continue

        files = os.listdir(class_path)

        for img_name in files:

            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(class_path, img_name)

            img = cv2.imread(img_path)

            if img is None:
                print(f"❌ Skipping: {img_path}")
                continue

            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img / 255.0

            data.append(img)
            labels.append(label_name)

    print(f"✅ Loaded {len(data)} images")

    data = np.array(data).reshape(-1, 224, 224, 1)
    labels = np.array(labels)

    return data, labels


# Load dataset
X_train, y_train = load_class_folders(TRAIN_PATH)
X_test, y_test = load_class_folders(TEST_PATH)

# Safety check
if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("❌ Dataset is empty!")

# -----------------------------
# LABEL ENCODING
# -----------------------------
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
class_names = le.classes_

print("Classes:", class_names)

# -----------------------------
# MODEL
# -----------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,1)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# TRAINING
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# -----------------------------
# PREDICTIONS
# -----------------------------
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# -----------------------------
# METRICS
# -----------------------------
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=class_names))

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
accuracy = np.mean(y_pred == y_test)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# ACCURACY GRAPH
# -----------------------------
plt.figure()
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.legend()
plt.title("Accuracy")
plt.show()

# -----------------------------
# PRECISION-RECALL CURVE
# -----------------------------
y_test_bin = tf.keras.utils.to_categorical(y_test, num_classes=len(class_names))

plt.figure()
for i in range(len(class_names)):
    p, r, _ = precision_recall_curve(y_test_bin[:, i], y_pred_probs[:, i])
    plt.plot(r, p, label=class_names[i])

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()