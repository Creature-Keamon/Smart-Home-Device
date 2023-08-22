import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model

#predicting a numerical value based on some other combination of variables

#creating scalar features (input factors affecting the outcome)
X = tf.range(-100, 100, 4)

#create labels (Y axis)
y = X + 10

#creating test and training data
X_train = X[:40] #first 40 numbers from X becomes the training data (80%)
X_test = X[40:]#last 10 numbers from X becomes testing data (20%)
y_train = y[:40] #same as above but for y
y_test = y[40:] #same as above but for y

# 1 create the model (this one can build automatically by setting it's shape)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1], name="input"),
    tf.keras.layers.Dense(1, input_shape=[1], name="middle_man"),
    tf.keras.layers.Dense(1, input_shape=[1], name="output")], name = 'hello.guys.its.me')

# 2 compile the model
model.compile(loss=tf.keras.losses.mae,
              optimizer= tf.keras.optimizers.SGD(),
              metrics=["mae"])

# 3 fit the model
model.fit(X_train, y_train, epochs=100, verbose=1)

#view the summary of the model
model.summary()

#creates a summary as a jpg file  
tf.keras.utils.plot_model(model=model, to_file = 'test_model.jpg', show_shapes =True)