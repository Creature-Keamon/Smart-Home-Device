#Going to cover some of the most fundamental concepts of tensors using Tensorflow

## CREATING DIFFERENT TENSORS

import tensorflow as tf

#creates tensor
scalar = tf.constant(7)
print(scalar)

#check number of dimensions of a tensor
scalar.ndim

#create a vector
vector = tf.constant([10,10])
print(vector)

vector.ndim

#create a matrix
matrix = tf.constant([[10,7],[7,10]])

matrix.ndim

another_matrix = tf.constant([[46.,5], [1.,2.], [2.,3.]], dtype=tf.float16) #specify datatype

another_matrix.ndim

tensor = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]])

print(tensor)

tensor.ndim
