from __future__ import print_function
import tensorflow as tf
import json
import numpy as np

# Load and preprocess the data from JSON file
def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    sequences = []
    labels = []
    label_map = {"Frisked": 1, "Not Frisked": 0}
    for item in data:
        sequence = item["keypoints_sequence"]
        # Ensure the sequence has the correct length and format
        if len(sequence) == 17 and all(len(point) == 3 for point in sequence):
            sequences.append(sequence)
            labels.append(label_map[item["label"]])
    return np.array(sequences), np.array(labels)

# Training parameters
learning_rate = 0.00001
training_steps = 250
batch_size = 16
display_step = 4

# Network parameters
num_input = 3  # 3 values per keypoint (x, y, confidence)
timesteps = 17  # 17 keypoints
num_hidden = 128  # Number of features in the hidden layer
num_classes = 2  # Frisked or Not Frisked

# Load data
sequences, labels = load_data('data.json')
labels = tf.keras.utils.to_categorical(labels, num_classes)

# Split data into training and test sets
split_index = int(len(sequences) * 0.8)
train_sequences, test_sequences = sequences[:split_index], sequences[split_index:]
train_labels, test_labels = labels[:split_index], labels[split_index:]

# Create a tf.data dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Define the RNN model
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

model = RNNModel(num_hidden, num_classes)

# Define loss and optimizer
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Define metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(inputs, labels):
    predictions = model(inputs, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

# Training loop
losses = []
accuracies = []

for epoch in range(training_steps):
    for batch, (inputs, labels) in enumerate(train_dataset):
        train_step(inputs, labels)
    
    if (epoch + 1) % display_step == 0:
        losses.append(float(train_loss.result()))
        accuracies.append(float(train_accuracy.result()))
        print(f"Epoch {epoch + 1}, Loss: {train_loss.result():.4f}, Accuracy: {train_accuracy.result():.4f}")

    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()

print("Optimization Finished!")

# Save the loss and accuracy data to a file
with open('training_data.json', 'w') as f:
    json.dump({'losses': losses, 'accuracies': accuracies}, f)

# Evaluate the model on the test set
test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_labels)).batch(batch_size)

for inputs, labels in test_dataset:
    test_step(inputs, labels)

print(f"Testing Accuracy: {test_accuracy.result():.4f}")

# Save the model
model.save_weights('rnn_model_weights')
print("Model weights saved.")