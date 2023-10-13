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
import random
import json

#defines lists
labels_str = []
labels_str2 = []
message = []
message2 = []
labels_int = []

#loads dataset
with open('full_spam_set.csv', 'r', encoding='utf-8') as dataset:
    csv_reader = csv.reader(dataset)
    
    #splits dataset into two lists, message 
    for line in csv_reader:
        message.append(line[0])
        labels_int.append(line[1])

labels_int = tf.convert_to_tensor(np.array(labels_int), dtype=tf.int32)

#prepares data to be used in the neural network
X_train = message[0:8960]
y_train = labels_int[0:8960]

X_test = message[8960:11200]
y_test = labels_int[8960:11200]

X_pred = message[11286:11287] # target = 1
X2_pred = message[11288:11289] #target = 0
X3_pred = ["""Ur HMV Quiz cash-balance is currently $500 - 
           to maximize ur cash-in now send HMV1 to 86688 only 150p/msg"""] # target = 1 (AI has had particular difficulty with this one)
y_pred = [0,0]

#generates tokenizer and configures it
tokenizer = Tokenizer(num_words= 20, oov_token="<OOV>")

#tokenises the text (assigns numerical values to every word)
tokenizer.fit_on_texts(X_train)

tokenizer.fit_on_texts(X_test) 

tokenizer.fit_on_texts(X_pred) 
tokenizer.fit_on_texts(X2_pred) 
tokenizer.fit_on_texts(X3_pred) 

#creates internal knowledge about each word and their values
word_index = tokenizer.word_index

#creates sequence of numbers which correlate to the words in the line
training_sequences = tokenizer.texts_to_sequences(X_train)

testing_sequence = tokenizer.texts_to_sequences(X_test)

prediction_sequence = tokenizer.texts_to_sequences(X_pred)
prediction_sequence2 = tokenizer.texts_to_sequences(X2_pred)
prediction_sequence3 = tokenizer.texts_to_sequences(X3_pred)

#adds '0s' to the end of the sequences to ensure that they are all equal length
training_padded = pad_sequences(training_sequences, padding='post')

testing_padded = pad_sequences(testing_sequence, padding='post')

#creates arrays for every value that will be used
training_padded = np.array(training_padded)
training_labels = np.array(y_train)

testing_padded = np.array(testing_padded)
testing_labels = np.array(y_test)

prediction_array = np.array(prediction_sequence)
prediction_array2 = np.array(prediction_sequence2)
prediction_array3 = np.array(prediction_sequence3)


#creating the model(s)
model1 = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 20),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(100, input_shape=[1],activation='relu'),
    tf.keras.layers.Dense(200, input_shape=[1],activation='relu'),
    tf.keras.layers.Dense(1, input_shape=[1], activation='sigmoid')
])

#view the summary of the model
model1.summary()

#compiles the model (establishes how to tell the model how wrong it is)
model1.compile(loss='binary_crossentropy',
                optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=["accuracy"])

#trains the model on the data I have fed it
model1.fit(training_padded, 
           training_labels, 
           epochs=200, 
           validation_data=(testing_padded, 
                            testing_labels))

#define the graphing plot function
def plot_predictions(prediction_values, true_label):
    plt.figure(figsize=(6,2.5)) #sets window size
    plt.scatter(prediction_values, true_label, c="g", label= "prediction") #places point on the graph
    plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]) #sets graph to show between 0 and 1
    for i, txt in enumerate(prediction_values):
        plt.annotate(txt, ((prediction_values[(i)]), (y_pred[(i)]))) # shows the point's value on the graph
    plt.legend();
    plt.show()

#predict model performance
predicted_sequence = model1.predict(prediction_array)
predictions = [1, predicted_sequence]
print(predictions)
#run the grapher
plot_predictions(predictions, y_pred)

predicted_sequence2 = model1.predict(prediction_array2)
predictions2 = [0, predicted_sequence2]
print(predictions2)
plot_predictions(predictions2, y_pred)

predicted_sequence3 = model1.predict(prediction_array3)
predictions3 = [1, predicted_sequence3]
print(predictions3)
plot_predictions(predictions3, y_pred)

#determines the next action based on if the user typed "Y", "N" or something else
while True:
    #asks user if they want to save
    plot_input = input("Do you wish to save the model with it's current performance? Y/N").upper()
    
    if plot_input == "Y":
        #Saving the model
        model1.save("spam_detection_model.keras")
        print("model saved")
        exit()
    
    
    elif plot_input == "N":
        exit()
    
    else:
        print("Try again")