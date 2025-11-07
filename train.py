import tensorflow as tf
from data_loader import create_datasets, BATCH_SIZE
from model_builder import build_model
import matplotlib.pyplot as plt
import os

# Define key parameters for the training process.
DATA_DIR = './data'
MODEL_SAVE_PATH = './models/waste_classifier_model.keras'
EPOCHS = 20 

# Load and prepare data
try:
    train_ds, validation_ds, test_ds, class_names = create_datasets(DATA_DIR)
    NUM_CLASSES = len(class_names)
except FileNotFoundError:
    print(f"Error: The data directory '{DATA_DIR}' was not found.")
    print("Please ensure your data is structured correctly inside the 'data' folder.")
    exit()

# Build the model
model = build_model(num_classes=NUM_CLASSES)

# compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
    )

# Train the model
print("\n Starting Model Training...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data = validation_ds
)
print('Model Training Finished.')

# Save the trained model
print(f"\nSaving trained model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("Model Saved Successfully")

#  Visualize Training History 
print("Generating training history plots...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('training_history.png')
plt.show()

print("\nScript finished. Your trained model is saved, and a plot of the training history has been generated as 'training_history.png'.")