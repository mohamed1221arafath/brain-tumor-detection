import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

IMG_SIZE = (150, 150)

# Load your trained model
model = tf.keras.models.load_model("brain_tumor_model.keras")

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Pick a sample image to test
img_path = os.path.join("Testing", "glioma", os.listdir("Testing/glioma")[15])

# Load and preprocess the image
img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Manually apply EfficientNet preprocessing (since we're bypassing the outer model)
preprocessed_img = tf.keras.applications.efficientnet.preprocess_input(img_array.copy())

# Get the base EfficientNet model and its last conv layer
base_model = model.get_layer("efficientnetb0")
last_conv_layer = base_model.get_layer("top_conv")

# Build a grad model using ONLY the base_model's own graph (avoids the nesting error)
grad_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=[last_conv_layer.output, base_model.output]
)

# Get the remaining layers (the "head" we added on top)
gap_layer = model.get_layer("global_average_pooling2d")
dense_layer = model.get_layer("dense")
dropout_layer = model.get_layer("dropout")
final_layer = model.get_layer("dense_1")

# Compute gradients
with tf.GradientTape() as tape:
    conv_outputs, base_output = grad_model(preprocessed_img)
    x = gap_layer(base_output)
    x = dense_layer(x)
    x = dropout_layer(x, training=False)
    predictions = final_layer(x)

    pred_index = tf.argmax(predictions[0])
    class_channel = predictions[:, pred_index]

grads = tape.gradient(class_channel, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)
heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
heatmap = heatmap.numpy()

# Overlay heatmap on original image
img_orig = cv2.imread(img_path)
img_orig = cv2.resize(img_orig, IMG_SIZE)

heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

superimposed = cv2.addWeighted(img_orig, 0.6, heatmap_colored, 0.4, 0)

predicted_class = class_names[pred_index.numpy()]
confidence = float(predictions[0][pred_index]) * 100

# Show side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original MRI")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Grad-CAM\nPredicted: {predicted_class} ({confidence:.1f}%)")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("gradcam_result.png")
plt.show()

print(f"Predicted: {predicted_class} ({confidence:.1f}% confidence)")
print("Saved as gradcam_result.png")