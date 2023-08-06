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

#indexing tensors

rank_4_tensor[:2, :2, :2, :2]

rank_4_tensor[:2, :2, :, :2]

rank_2_tensor = tf.constant(shape=[[2, 3], [4, 5]])

#get last item of each rank 2 tensor
rank_2_tensor[:, -1]

#add an extra dimension to the tensor making it larger
rank_3_tensor = rank_2_tensor[..., tf.newaxis]

#alternative

tf.expand_dims(rank_2_tensor, axis=-1)

tensor = tf.constant([[10, 7], [3, 4]])

tensor + 10 # adds 10 to every element in the tensor
tf.add(tensor, 10) # does the same


#matrix multiplication

matrix_1 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix_2 = tf.constant([[4, 78, 54], [4, 1, 6], [7, 9, 32]])
tf.matmul(matrix_1, matrix_2)

tf.reshape(matrix_1, shape=(3, 3))

#visualise data
plt.figure(figsize=(10, 7))

#plot training data
plt.scatter(X_train, y_train, c="g", label="Training data")

#plot testing data
plt.scatter(X_test, y_test, c="r", label="Testing data")

plt.show()