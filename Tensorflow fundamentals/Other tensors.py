import tensorflow as tf
import numpy as np

print(tf.ones([10, 7]))

print(tf.zeros([3, 7]))

numpy_A = np.arange(1, 25, dtype=np.int32)

X = tf.constant(numpy_A, shape=(2 ,3 ,4))

print(X) 

rank_4_tensor = tf.zeros(shape=[2, 3, 4, 5 ])

print("datatype of every element", rank_4_tensor.dtype)
print("Number of dimensions(rank):", rank_4_tensor.ndim)
print("shape of tensor", rank_4_tensor.shape)
print("elements along the 0 axis", rank_4_tensor.shape[0])
print("elements along the last axis", rank_4_tensor.shape[-1])
print("total number of elements", tf.size(rank_4_tensor).numpy())