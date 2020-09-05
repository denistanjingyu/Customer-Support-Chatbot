"""Chatbot model dataset preparation."""

# Set working directory
import os
os.chdir('C:\\Users\\user\\Desktop\\Chatbot Project')

# Import libraries and NLP Preprocessing Pipeline
from nltk_utils import tokenize, stemming, bag_of_words
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
import json
import numpy as np
import torch
import torch.nn as nn

# Load JSON file containing user intent and chatbot reply
with open('tags.json', 'r') as f:
    intents = json.load(f)

# Create an empty list which stores all the processed words
all_words = []
# Create an empty list which stores all the tags
tags = []
# Create an empty list which stores all the patterns and responses
xy = []

# Loop through corpus and applying preprocessing techniques
# Retrieve tags
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    # Retrieve processed word and tag
    for pattern in intent['patterns']:
        # Tokenize and add to list
        word = tokenize(pattern)
        all_words.extend(word)
        xy.append((word, tag))

# Define list of words to ignore
ignore_words = ['?', '!', '.', "'s"]

# Apply stemming function and ignore the punctuations
all_words = [stemming(word) for word in all_words if word not in ignore_words]

# Sort all the words and take only the unique words
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create list to store training data
X_train = []
y_train = []

# Add bag of words to X training data and tag to y training data
for (pattern_sentence, tag) in xy:
    # Convert pattern to bag of words
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # Convert tag to number
    label = tags.index(tag)
    y_train.append(label)

# Convert training data to numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    """Define and store the attributes of the training dataset."""

    def __init__(self):
        """Create dataset for X_train and y_train."""
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        """Return element at index k."""
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        """Return length of data."""
        return self.n_samples


# Hyperparameters input values
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
n_epochs = 1500

# Instantiate the class ChatDataset and DataLoader
dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset,
                          batch_size = batch_size,
                          shuffle = True,
                          num_workers = 0)

# Instantiate the NeuralNet class
model = NeuralNet(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Training loop
for epoch in range(n_epochs):
    for (words, labels) in train_loader:
        # Forward pass
        labels = labels.long()
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward pass and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss every 100 epoch
    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch + 1} / {n_epochs}, loss = {loss.item():.4f}')

# Print the final loss
print(f'Final loss, loss = {loss.item():.4f}')

# Save the model
data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
        }

FILE = "data.pth"
torch.save(data, FILE)

# Output completion status
print(f'Training complete. File saved to {FILE}.')
