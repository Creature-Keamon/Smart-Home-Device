import tensorflow as tf
import csv
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np

labels_str = []
message = []
labels_int = []

#loads dataset
with open('spam.csv', 'r') as dataset:
    csv_reader = csv.reader(dataset)
    
    #splits dataset into two lists
    for line in csv_reader:
        labels_str.append(line[0])
        message.append(line[1])
    
    #creates a new list and adds 1s and 0s to it based on "ham"s and "spam"s in labels
    for line in labels_str:
        if "ham" in line:
                labels_int.append(1)
        elif "spam" in line:
                labels_int.append(0)

#visualising data to ensure that the values in labels_int match with the message list
message_list = list(message)
for x in range(len(labels_int)):
    print(labels_int[x], message_list[x])

#prepares data to be used in the neural network
X_train = message[:4460]
X_test = message[4460:]
y_train = labels_int[:4460]
y_test = labels_int[4460:]

#generates tokenizerand configures it
tokenizer = Tokenizer(num_words= 20, oov_token="<OOV>")

#creates internal knowledge about vocabulary
tokenizer.fit_on_texts(X_train)

#tokenises the test (assigns numerical values to every word)
word_index = tokenizer.word_index


training_sequences = tokenizer.texts_to_sequences(X_train)