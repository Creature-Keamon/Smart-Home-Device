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
    
    #creates a new list and adds 1s and 0s to it based on "ham"s and "spam"s in labels_str respectively
    for line in labels_str:
        if "ham" in line:
                labels_int.append(1)
        elif "spam" in line:
                labels_int.append(0)

labels_int = tf.convert_to_tensor(np.array(labels_int), dtype=tf.int32)

#prepares data to be used in the neural network
X_train = message[1:4460]
X_test = message[4460:5573]
y_train = labels_int[1:4460]
y_test = labels_int[4460:5573]
X_pred = ["reply to this message to recieve a free $200 iTunes gift card!"]
y_pred = 0

#generates tokenizerand configures it
tokenizer = Tokenizer(num_words= 20, oov_token="<OOV>")

#creates internal knowledge about vocabulary
tokenizer.fit_on_texts(X_train)
tokenizer.fit_on_texts(X_test)
tokenizer.fit_on_texts(X_pred) 

#tokenises the test (assigns numerical values to every word)
word_index = tokenizer.word_index

#creates sequence of numbers which correlate to the words in the line
training_sequences = tokenizer.texts_to_sequences(X_train)
prediction_sequence = tokenizer.texts_to_sequences(X_pred)
testing_sequence = tokenizer.texts_to_sequences(X_test)

#adds '0s' to the end of the sequences to ensure that they are all equal length
training_padded = pad_sequences(training_sequences, padding='post')
testing_padded = pad_sequences(testing_sequence, padding='post')

#creates arrays for every value that will be used
training_padded = np.array(training_padded)
training_labels = np.array(y_train)
testing_padded = np.array(testing_padded)
testing_labels = np.array(y_test)
prediction_array = np.array(prediction_sequence)

#creating the model(s)
model1 = tf.keras.Sequential([
    tf.keras.layers.Embedding(999, 20),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(10, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#view the summary of the model
model1.summary()

#compiles the model (tells the model how wrong it is)
model1.compile(loss=tf.keras.losses.mae,
               optimizer=tf.keras.optimizers.SGD(),
               metrics=["mae"])
#trains the model on the data I have fed it
model1.fit(training_padded, training_labels, epochs=100)

#predict model performance
predicted_sequence1 = model1.predict(prediction_array)

#calculate mean absolute error and square errors of predictions
mae_1 = keras.losses.MAE(y_test, predicted_sequence1)
mse_1 = keras.losses.MSE(y_test, predicted_sequence1)

#define the graphing plot function
def plot_predictions(prediction, true_label):
    plt.figure(figsize=(6,2.5)) #sets window size
    plt.scatter(prediction, y_pred, c="g", label= "prediction") #places point on the graph
    plt.xticks(np.arange(0, 1, step=0.2)) #sets graph to show between 0 and 1
    for i, txt in enumerate(prediction):
        plt.annotate(txt, ((prediction-0.2), (y_pred+0.005))) # shows the point's value on the graph
    plt.legend();
    plt.show()

#run the grapher
plot_predictions(predicted_sequence1, y_pred)

#determines the next action based on if the user typed "Y", "N" or something else
while True:
    #asks user if they want to save
    plot_input = input("Do you wish to save the model with it's current performance? Y/N").upper()
    
    if plot_input == "Y":
        #Saving the model
        model1.save("spam_detection_model")
        model1.save("spam_detection_model.h5")
        print("model saved as SavedModel and also HDF5")
        exit()
    
    
    elif plot_input == "N":
        print("Alright Then")
        exit()
    
    else:
        print("Try again")