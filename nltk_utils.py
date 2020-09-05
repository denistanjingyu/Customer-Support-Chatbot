"""NLP Preprocessing Pipeline."""

# Import libraries
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

# Instantiate stemmer
stemmer = PorterStemmer()


# Tokenize function
def tokenize(sentence):
    """Take a sentence and break it into individual linguistic units.

    :param: single sentence
    :type: string
    :rtype: list
    :return: list of words
    """
    return nltk.word_tokenize(sentence)


# Stemming function
def stemming(word):
    """Take a word, convert to lower case and remove the suffix.

    :param: single word
    :type: string
    :rtype: string
    :return: stemmed word
    """
    return stemmer.stem(word.lower())


# Bag-of-words model function
def bag_of_words(tokenized_sentence, words):
    """Take a tokenized sentence, apply stemming and convert to bag of words.

    :param: single word
    :type: string
    :rtype: list
    :return: list of binary numbers
    """
    # Apply stemming function to the words
    sentence_words = [stemming(word) for word in tokenized_sentence]

    # Create an array to store the bag of words
    bag = np.zeros(len(words), dtype = np.float32)

    # Form the bag of words with 1 or 0
    for index, word in enumerate(words):
        if word in sentence_words:
            bag[index] = 1

    return bag
