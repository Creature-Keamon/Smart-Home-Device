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

model2 = tf.keras.Sequential([
    tf.keras.layers.Embedding(999, 20),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model3 = tf.keras.Sequential([
    tf.keras.layers.Embedding(999, 20),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#view the summary of the model
model1.summary()

#compiles the model (tells the model how wrong it is)
model1.compile(loss=tf.keras.losses.mae,
               optimizer=tf.keras.optimizers.SGD(),
               metrics=["mae"])

model2.compile(loss=tf.keras.losses.mae,
               optimizer=tf.keras.optimizers.SGD(),
               metrics=["mae"])

model3.compile(loss=tf.keras.losses.mae,
               optimizer=tf.keras.optimizers.SGD(),
               metrics=["mae"])

#trains the model on the data I have fed it
model1.fit(training_padded, training_labels, epochs=100)
model2.fit(training_padded, training_labels, epochs=100)
model3.fit(training_padded, training_labels, epochs=500)

#predict model performance
predicted_sequence1 = model1.predict(prediction_array)
predicted_sequence2 = model2.predict(prediction_array)
predicted_sequence3 = model3.predict(prediction_array)

print(predicted_sequence1, predicted_sequence2, predicted_sequence3)

#calculate mean absolute error and square errors of predictions
mae_1 = keras.losses.MAE(y_test, predicted_sequence1)
mse_1 = keras.losses.MSE(y_test, predicted_sequence1)
mae_2 = keras.losses.MAE(y_test, predicted_sequence2)
mse_2 = keras.losses.MSE(y_test, predicted_sequence2)
mae_3 = keras.losses.MAE(y_test, predicted_sequence3)
mse_3 = keras.losses.MSE(y_test, predicted_sequence3)

#prepare results into a nice list
model_results = [["model 1", mae_1.numpy(), mse_1.numpy()],
                 ["model 2", mae_2.numpy(), mse_2.numpy()],
                 ["model 3", mae_3.numpy(), mse_3.numpy()]]

#put the list into a table
all_results = pd.DataFrame(model_results, columns = ["model", "mae", "mse"])

# print table
print(all_results)

#define the graphing plot function
def plot_predictions(prediction, true_label):
    plt.figure(figsize=(6,2.5))
    plt.scatter(prediction, y_pred, c="g", label= "prediction")
    plt.xticks(np.arange(0, 1, step=0.2))
    for i, txt in enumerate(prediction):
        plt.annotate(txt, ((prediction-0.2), (y_pred+0.005)))
    plt.legend();
    plt.show()

#run the grapher
plot_predictions(predicted_sequence, y_pred)

#predicting a user input of text
user_text = input("input your text")

user_sequence = tokenizer.texts_to_sequences(user_text)
user_array = np.array(user_sequence)
predict_user_sequence = model.predict(user_array)

plot_predictions(predict_user_sequence, y_pred)
