from cProfile import label
import tensorflow as tf
import csv
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

#creating the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(999, 20),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#view the summary of the model
model.summary()

#compiles the model (tells the model how wrong it is)
model.compile(loss=tf.keras.losses.mae,
               optimizer=tf.keras.optimizers.SGD(),
               metrics=["mae"])

#trains the model on the data I have fed it
model.fit(training_padded, training_labels, epochs=25)

#plot_predictions(y_pred, pred_true_label)
predicted_sequence = model.predict(prediction_sequence)

#define the graphing plot function
def plot_predictions(prediction, true_label):
    plt.figure(figsize=(10,10))
    plt.scatter(prediction, y_pred, c="g", label= "prediction")
    plt.xticks(np.arange(-0.1, 1.1, step=0.2))
    plt.legend();
    plt.show()

#run the grapher
plot_predictions(predicted_sequence, y_pred)
