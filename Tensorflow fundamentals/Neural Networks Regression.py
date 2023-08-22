import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#predicting a numerical value based on some other combination of variables

#creating scalar features (input factors affecting the outcome)
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

#create labels (Y axis)
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

#converting np array into tensor
Xtensor = tf.cast(tf.constant(X), dtype=tf.float32)
ytensor = tf.cast(tf.constant(y), dtype=tf.float32)

input_shape = X[0].shape
output_shape = y[0].shape

#create a seed
tf.random.set_seed(42)

#Step 1. creating a model
model = tf.keras.Sequential([ #"provides training and inference features for the model"
    tf.keras.layers.Dense(100, activation ="relu"),
    tf.keras.layers.Dense(1)
    ])

#Step 2. Compile the model
model.compile(loss=tf.keras.losses.mae, #mae = mean absolute error
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), #SGD = stochastic gradient descent (tells model how it should improve)
              metrics=["mae"])

#Step 3. fit the model
model.fit(tf.expand_dims(X, axis=-1), y, epochs=100) #makes model figures out the relationship between X and y

#make a prediction with the model (what value of y does it think will be when X is 17)
y_prediction = model.predict([17.0])
print(y_prediction)
