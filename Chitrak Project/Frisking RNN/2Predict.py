import tensorflow as tf
import json
import numpy as np
import os

# Function to load and preprocess the JSON file with keypoints sequences
def load_json_file(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    sequences = [item for item in data if len(item) == 17]
    return np.array(sequences)

# Load the RNN model
class RNNModel(tf.keras.Model):
    def __init__(self, num_hidden, num_classes):
        super(RNNModel, self).__init__()
        self.lstm = tf.keras.layers.LSTM(num_hidden, return_sequences=False)
        self.dropout = tf.keras.layers.Dropout(0.5)  # Add dropout layer
        self.dense = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        x = self.lstm(x)
        x = self.dropout(x)
        return self.dense(x)

num_hidden = 128
num_classes = 2
model = RNNModel(num_hidden, num_classes)
model.load_weights('rnn_model_weights')

# Load keypoints sequences from the JSON file
sequences = load_json_file('output_frames/keypoints.json')

# Make predictions using the model
predictions = model.predict(sequences)

# Convert predictions to labels based on threshold (assuming class 1 is "Frisked")
labels = np.argmax(predictions, axis=1)  # 0 for Not Frisked, 1 for Frisked

# Create directories for saving the results
output_dir = 'output'
frisked_dir = os.path.join(output_dir, 'frisked')
not_frisked_dir = os.path.join(output_dir, 'not_frisked')

os.makedirs(frisked_dir, exist_ok=True)
os.makedirs(not_frisked_dir, exist_ok=True)

# Save sequences based on predictions
for i, label in enumerate(labels):
    sequence = sequences[i].tolist()
    if label == 1:
        with open(os.path.join(frisked_dir, f'frisked_sequence_{i}.json'), 'w') as f:
            json.dump(sequence, f)
    else:
        with open(os.path.join(not_frisked_dir, f'not_frisked_sequence_{i}.json'), 'w') as f:
            json.dump(sequence, f)

# Count predictions
total_sequences = len(predictions)
frisked_count = np.sum(labels)  # Count of sequences labeled as Frisked

# Calculate percentage of frisked sequences
frisked_percentage = frisked_count / total_sequences

# Print results
print(f"Total sequences: {total_sequences}")
print(f"Number of sequences labeled as Frisked: {frisked_count}")
print(f"Number of sequences labeled as Not Frisked: {total_sequences - frisked_count}")

# Final prediction based on frisked percentage
if frisked_percentage > 0.3:
    print("Final Prediction: Frisked")
else:
    print("Final Prediction: Not Frisked")
