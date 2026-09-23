import tensorflow as tf

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# Load training data
train_data = tf.keras.utils.image_dataset_from_directory(
    "Training",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# Load testing data
test_data = tf.keras.utils.image_dataset_from_directory(
    "Testing",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# Print class names to confirm
print("Classes found:", train_data.class_names)

# Normalize pixel values (0-255 -> 0-1)
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
test_data = test_data.map(lambda x, y: (normalization_layer(x), y))

print("Data is ready for training!")