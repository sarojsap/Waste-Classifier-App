# evaluate.py

import tensorflow as tf
from data_loader import create_datasets


DATA_DIR = './data'
MODEL_PATH = './models/waste_classifier_model.keras'

# Load the Test Data 
print("Loading test dataset...")
try:
    _, _, test_ds, class_names = create_datasets(DATA_DIR)
except FileNotFoundError:
    print(f"Error: The data directory '{DATA_DIR}' was not found.")
    exit()

# Load the Trained Model 
# We load the model that we saved at the end of the training script.
print(f"Loading trained model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except (ImportError, IOError) as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model file exists and was saved correctly.")
    exit()

#Evaluate the Model 
# The model.evaluate() function computes the loss and accuracy of the model on the provided dataset.
print("\n--- Evaluating model on the test set ---")
loss, accuracy = model.evaluate(test_ds)

print("\nEvaluation Results: ")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f} ({accuracy:.2%})")