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
    ])

# 2 compile the model
model.compile(loss=tf.keras.losses.mae,
              optimizer= tf.keras.optimizers.SGD(),
              metrics=["mae"])

# 3 fit the model
model.fit(X_train, y_train, epochs=500, verbose=1)

#view the summary of the model
model.summary()

#creates a summary as a jpg file  
tf.keras.utils.plot_model(model=model, to_file = 'test_model.jpg', show_shapes =True)

#make the model predict what the y vales will be compared to X
y_pred = model.predict(X_test)

#define the plot function
def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions= y_pred):
    plt.figure(figsize=(7,4))
    #plot training data
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    #plot testing data
    plt.scatter(test_data,test_labels, c="g", label= "Testing data")
    #plot prediction vs testing
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    plt.legend();
    plt.show()
    
plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions= y_pred)

#evaluating the model on test data, OUTPUT(<loss function>, <evaluation metric>)
model.evaluate(X_test, y_test)

#calculate the mean absolute error (MAE) between test and predictions
mae = tf.keras.losses.MAE(y_true=y_test, y_pred=tf.squeeze(y_pred))

#calculate the mean square error (MSE) between test and predictions
mse = tf.keras.losses.MSE(y_true=y_test, y_pred=tf.squeeze(y_pred))

