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
        print("Your message is likely to be legitimate")
    elif prediction > 0.5 and prediction < 0.75:
        print("Your message is likely to be spam")
    elif prediction > 0.75 and prediction < 1:
        print("Your message is very likely to be spam!")

#load in the model
model_load = tf.keras.models.load_model("spam_detection_model")

model_load.compile(loss='binary_crossentropy',
                optimizer= tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

tokenizer = Tokenizer(num_words= 300, oov_token="<OOV>")

#gets user input
user_input = input("Type your message that you want to be checked for spam")

user_list.append(user_input)

#preprocesses the user input into tensors
tokenizer.fit_on_texts(user_list)
word_index = tokenizer.word_index

user_sequence = tokenizer.texts_to_sequences(user_list)
user_array = np.array(user_sequence)



#makes prediction about the user's input
user_predict = model_load.predict(user_array)

#runs grapher
plot_predictions(user_predict, y_value)

