from cProfile import label
import tensorflow as tf
from tensorflow import keras
from keras.losses import MeanAbsoluteError, MeanSquaredError
import csv
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np
import json

user_list = []
y_value = [0]
labels_str = []
labels_str2 = []
message = []
message2 = []
labels_int = []

with open('full_spam_set.csv', 'r', encoding='utf-8') as dataset:
    csv_reader = csv.reader(dataset)
    
    #splits dataset into two lists
    for line in csv_reader:
        message.append(line[0])
        labels_int.append(line[1])

labels_int = tf.convert_to_tensor(np.array(labels_int), dtype=tf.int32)

#prepares data to be used in the neural network
X_test = message[5574:9043]
y_test = labels_int[5574:9043]

  #define the graphing plot function
def plot_predictions(prediction, y_value):
    plt.figure(figsize=(6,2.5)) #sets window size
    plt.scatter(prediction, y_value, c="g", label= "prediction") #places point on the graph
    plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]) #sets graph to show between 0 and 1
    for i, txt in enumerate(prediction):
        plt.annotate(txt, ((prediction[(i)]), (y_value[(i)]))) # shows the point's value on the graph
    plt.legend();
    plt.show()

def user_output(prediction):
    if prediction < 0.25:
        print("Your message is very likely to be legitimate!")
    elif prediction > 0.25 and prediction < 0.5:
        print("Your message is probably legitimate")
    elif prediction > 0.5 and prediction < 0.75:
        print("Your message is probably spam")
    elif prediction > 0.75 and prediction < 1:
        print("Your message is very likely to be spam!")

#load in the model
model_load = tf.keras.models.load_model("spam_detection_model.keras")

model_load.summary()

tokenizer = Tokenizer(num_words= 300, oov_token="<OOV>")

user_input = input("Paste your message here you want to be tested")
user_list.append(user_input)
print(user_list)

#preprocesses the user input into tensors
tokenizer.fit_on_texts(user_list)
tokenizer.fit_on_texts(X_test) 
word_index = tokenizer.word_index
user_sequence = tokenizer.texts_to_sequences(user_list)
testing_sequence = tokenizer.texts_to_sequences(X_test)
testing_padded = pad_sequences(testing_sequence, padding='post')
testing_padded = np.array(testing_padded)
testing_labels = np.array(y_test)
print(user_sequence)
user_array = np.array(user_sequence)
print(user_array)

model_load.compile(loss='binary_crossentropy',
                  optimizer= tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

model_load.evaluate(testing_padded, testing_labels)

#makes prediction about the user's input
user_predict = model_load.predict(user_array)
print(user_predict)

user_output(user_predict)

#runs grapher
plot_predictions(user_predict, y_value)

