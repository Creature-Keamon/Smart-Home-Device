import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
 ])

# 2 compile the model
model.compile(loss=tf.keras.losses.mae,
              optimizer= tf.keras.optimizers.SGD(),
              metrics=["mae"])

model.summary()

# 3 fit the model
#model.fit(X_train, y_train, epochs=100)
