"""Chatbot program."""

# Import libraries
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
import random
import json
import torch

# Load json file
with open('tags.json', 'r') as f:
    intents = json.load(f)

# Load data
FILE = "data.pth"
data = torch.load(FILE)

# Extract data to variables
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Load model parameters
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Create bot
# Define a bot name
bot_name = "Atlas"
# Welcome message and exit instruction
print("Welcome here! Type 'quit' to exit.")
while True:
    # Get input from user
    sentence = input("You: ")
    # Exit program if quit entered
    if sentence == "quit":
        break

    # Otherwise preprocess sentence
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    # Get prediction from model
    output = model(X)
    _, predicted = torch.max(output, dim = 1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim = 1)
    prob = probs[0][predicted.item()]

    # Probability threshold for response
    if prob.item() > 0.5:
        # Check for a matching tag
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
