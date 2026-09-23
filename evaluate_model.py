import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# Load the saved model
model = tf.keras.models.load_model("brain_tumor_model.keras")

# Load test data (must match training setup exactly)
test_data = tf.keras.utils.image_dataset_from_directory(
    "Testing",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = test_data.class_names
print("Classes:", class_names)

# Get true labels
y_true = np.concatenate([y for x, y in test_data], axis=0)
y_true = np.argmax(y_true, axis=1)

# Get predictions
y_pred_probs = model.predict(test_data)
y_pred = np.argmax(y_pred_probs, axis=1)

# Print classification report (precision, recall, f1-score per class)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Build and plot confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Brain Tumor Classification")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("\nConfusion matrix saved as confusion_matrix.png")