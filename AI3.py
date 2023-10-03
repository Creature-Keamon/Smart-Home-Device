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
labels_str2 = []
message = []
message2 = []
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
                labels_int.append(0)
        elif "spam" in line:
                labels_int.append(1)
                
with open('emails.csv', 'r') as dataset2:
    csv_reader2 = csv.reader(dataset2)
    
    for line in csv_reader2:
        labels_int.append(line[1])
        message.append(line[0])

labels_int = tf.convert_to_tensor(np.array(labels_int), dtype=tf.int32)

#prepares data to be used in the neural network
X_train = message[0:11302]
y_train = labels_int[0:11302]
X_pred = message[11302:11303] # target = 0
X2_pred = message[34:35] #target = 1
y_pred = [0,0]

print (X_pred)
print(X2_pred)

#generates tokenizer and configures it
tokenizer = Tokenizer(num_words= 20, oov_token="<OOV>")

#creates internal knowledge about vocabulary
tokenizer.fit_on_texts(X_train)
tokenizer.fit_on_texts(X_pred) 
tokenizer.fit_on_texts(X2_pred) 

#tokenises the test (assigns numerical values to every word)
word_index = tokenizer.word_index

#creates sequence of numbers which correlate to the words in the line
training_sequences = tokenizer.texts_to_sequences(X_train)
prediction_sequence = tokenizer.texts_to_sequences(X_pred)
prediction_sequence2 = tokenizer.texts_to_sequences(X2_pred)

#adds '0s' to the end of the sequences to ensure that they are all equal length
training_padded = pad_sequences(training_sequences, padding='post')
#creates arrays for every value that will be used
training_padded = np.array(training_padded)
training_labels = np.array(y_train)
prediction_array = np.array(prediction_sequence)
prediction_array2 = np.array(prediction_sequence2)


#creating the model(s)
model1 = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 20),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(100, input_shape=[1]),
    tf.keras.layers.Dense(100, input_shape=[1]),
    tf.keras.layers.Dense(1, input_shape=[1])
])

#view the summary of the model
model1.summary()

#compiles the model (tells the model how wrong it is)
model1.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["mae"])
#trains the model on the data I have fed it
model1.fit(training_padded, training_labels, epochs=100)

#define the graphing plot function
def plot_predictions(prediction, true_label):
    plt.figure(figsize=(6,2.5)) #sets window size
    plt.scatter(prediction, y_pred, c="g", label= "prediction") #places point on the graph
    plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]) #sets graph to show between 0 and 1
    for i, txt in enumerate(prediction):
        plt.annotate(txt, ((prediction[(i)]), (y_pred[(i)]))) # shows the point's value on the graph
    plt.legend();
    plt.show()

#predict model performance
predicted_sequence = model1.predict(prediction_array)
predictions = [0, predicted_sequence]
#run the grapher
plot_predictions(predictions, y_pred)

predicted_sequence2 = model1.predict(prediction_array2)
predictions2 = [1, predicted_sequence2]
plot_predictions(predictions2, y_pred)

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