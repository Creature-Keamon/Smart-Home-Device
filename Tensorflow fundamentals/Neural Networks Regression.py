import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#predicting a numerical value based on some other combination of variables

#creating scalar features (input factors affecting the outcome)
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

#create labels
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

#visualise the graph as a scatter graph
plt.scatter(X,y);
plt.show()


#demo
house_info = tf.constant(["bedroom", "bathroom", "garage"])
house_price = tf.constant([939570])

#checking shape
input_shape = X[0].shape

X[0].ndim # check dimensions

#converting np array into tensor
Xtensor = tf.constant(X)
ytensor = tf.constant(y)

input_shape = X[0].shape
output_shape = y[0].shape