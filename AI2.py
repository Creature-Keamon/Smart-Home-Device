import tensorflow as tf
import csv
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np

labels_str = []
message = []
labels_int = []
training_padded_summarised = []
training_labels_summarised = []

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

labels_int = tf.constant(labels_int)

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

#creates sequence of numbers which correlate to the words in the line
training_sequences = tokenizer.texts_to_sequences(X_train)

#adds '0s' to the end of the sequences to ensure that they are all equal length
training_padded = pad_sequences(training_sequences,
                                padding='post')

#same as above but for testing instead
testing_sequences = tokenizer.texts_to_sequences(X_test)
testing_padded = pad_sequences(testing_sequences, 
                               padding='post')

#creates arrays for every value that will be used
training_padded = np.array(training_padded)
training_labels = np.array(y_train)
testing_padded = np.array(testing_padded)
testing_labels = np.array(y_test)

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

#make the model predict what the y vales will be compared to X
pred_sentence = "U won 100000 dolars claim now with this link right here"
sequences = tokenizer.texts_to_sequences(pred_sentence)
pred_padded = pad_sequences(sequences, padding="post")

print (pred_padded)


#define the plot function
def plot_predictions(prediction, true_label):
    plt.figure(figsize=(10,10))
    plt.scatter(prediction,true_label, c="g", label= "prediction data")
    
plot_predictions(pred_padded, 1)
#plot_predictions(y_pred, pred_true_label)
    
    





def black():
    def plot_predictions(test_data=testing_padded, test_labels=testing_labels, 
                         predictions= y_pred):
        plt.figure(figsize=(10,10))
        #plot testing data
        plt.scatter(test_data,test_labels, c="g", label= "Testing data")
        #plot prediction vs testing
        plt.scatter(test_data, predictions, c="r", label="Predictions")
        plt.legend();
        plt.show()
    
    plot_predictions(test_data=testing_padded, test_labels=testing_labels, 
                 predictions= y_pred)
