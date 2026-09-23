import tensorflow as tf

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# Load data
train_data = tf.keras.utils.image_dataset_from_directory(
    "Training",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

test_data = tf.keras.utils.image_dataset_from_directory(
    "Testing",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = train_data.class_names
print("Classes:", class_names)

# NOTE: No manual rescaling here — EfficientNet has its own preprocessing built in

# Use a pretrained model (transfer learning) as the base
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(150, 150, 3)
)
base_model.trainable = False  # freeze the pretrained layers

# Build the full model — preprocessing layer included INSIDE the model now
inputs = tf.keras.Input(shape=(150, 150, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=5
)

# Save the trained model
model.save("brain_tumor_model.keras")
print("\nModel saved as brain_tumor_model.keras")