import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#predicting a numerical value based on some other combination of variables

#creating scalar features (input factors affecting the outcome)
X = tf.range(-100, 100, 4)

#create labels (Y axis)
y = X + 10

#visualise data
plt.scatter(X, y)


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
model.fit(tf.expand_dims(X, axis=-1), y, epochs=150) #makes model figures out the relationship between X and y

#make a prediction with the model (what value of y does it think will be when X is 17)
y_prediction = model.predict([17.0])
print(y_prediction)
