# Customer Support Chatbot

![1_9I6EIL5NG20A8se5afVmOg](https://user-images.githubusercontent.com/45563371/92311742-7a378280-efec-11ea-9932-63d0b70495c9.gif)

## Project Objectives
1) Apply natural language processing concepts in the context of a chatbot
2) Implement an artificial neural network using PyTorch

## Overview
### Dataset Creation
Create a JSON file containing the intents of the customers
Metadata:
1) Patterns: How the customer will type and enquire
2) Responses: How the chatbot will respond
3) Tag: Single word to categorize the patterns

### Natural Language Processing Pipeline
Create a script to store the NLP techniques as functions
1) Tokenize: Take a sentence and break it into individual linguistic units
2) Stemming: Take a word, convert to lower case and remove the suffix
3) Bag-of-words model: Take a tokenized sentence, apply stemming and convert to bag of words

### Dataset Preparation
Application of natural language processing pipeline to dataset (corpus)
1) Load JSON file containing user intent
2) Create 3 lists to store processed words, tags, patterns/responses
3) Define list of words to ignore (mainly punctuation)
4) Loop through corpus and applying preprocessing techniques
5) Sort all the words and take only the unique words
6) Create 2 lists to store training data after preprocessing
    - Add bag of words to X training data 
    - Addtag to y training data
  
### Model Training
Define a simple artificial neural network model architecture using PyTorch
1) Define the layers and number of classes
    - 1 input layer, 1 hidden layer, 1 output layer
    - Activation function: rectified linear unit (ReLU)
    - 7 classes to predict
2) Define the hyperparameters
    - batch_size = 8
    - hidden_size = 8
    - learning_rate = 0.001
    - n_epochs = 1500
    - Optimizer = Adam
    - Loss = CrossEntropyLoss
    
  * Quite straightforward prediction and do not require further tuning
