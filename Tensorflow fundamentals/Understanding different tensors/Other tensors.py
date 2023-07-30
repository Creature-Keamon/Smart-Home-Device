import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

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

#reshaping
tf.reshape(rank_2_tensor, shape=(4, 1))

tf.expand_dims(rank_2_tensor, axis=-1)

tensor = tf.constant([[10, 7], [3, 4]])

tensor + 10 # adds 10 to every element in the tensor
tf.add(tensor, 10) # does the same

#transpose reshapes but swaps axis
tf.transpose(rank_3_tensor)

tf.matmul(tf.transpose(rank_2_tensor), rank_4_tensor)

#matrix multiplication is also called the dot product

#can also be done with tensordot
tf.tensordot(tf.transpose(rank_2_tensor), rank_4_tensor, axes=1)

#when performing matrix multiplication on two tensors and of the axes doesn't line up, you will transpose rather than reshape one of the tensors to satisfy the matrix multiplication


#create new tensor with default datatype
B = tf.constant([1.7, 2.4]) #float

C = tf.constant([10, 7]) #integer

B=tf.cast(B, dtype=tf.float16) #changes from 32bit to 16bit

C=tf.cast(C, dtype=tf.float32) #changes from integer to float

#aggregation is where tensors are condensed from many values down into a smaller amount of values

D = tf.constant([-7, -10])
tf.abs(D) #makes negatives into positives

E = tf.constant(np.random.randint(0,100, size=50))
print(tf.size(E), E.Shape, E.ndim)

tf.reduce_min(E) #finds smallest value in tensor

tf.reduce_max(E) #finds largest

tf.reduce_mean(E) #finds average

tf.reduce_sum(E) #finds sum of tensor

#finds variance (how far each number is from the average)
tf.math.reduce_variance(tf.cast(E, dtype=tf.float32))

#finds standard deviation (how dispersed each number is in relation to average)
tf.math.reduce_std(tf.cast(E, dtype=tf.float32))

#finding positional maxes and mins (The largest/smallest value in an index)
F = tf.random.uniform(shape=[50,3])

tf.argmax(F) #finds max value's location

F[tf.argmax(F)] #indexes largest value (finds value)

tf.argmin(F) #finds min location

#squeeze tensor (remove single dimensions)
G= tf.constant(tf.random.uniform(shape=[50]), shape=(1, 1, 1, 1, 50))

G_squeezed = tf.squeeze(G)

#one hot encoding (when only one row has a value for one of the columns)
some_list = [0, 1, 2, 3, 4]
tf.one_hot(some_list, depth= 5)

#specifying custom values for one hot encoding
tf.one_hot(some_list, depth= 5, on_value="what's goody gamers", off_value="it's me, keamon")

#square the tensor
F = tf.range(1, 10)
tf.square(F)

#square root
tf.sqrt(tf.cast(F, dtype=tf.float32))

#find the log of the tensor
tf.math.log(tf.cast(F, dtype=tf.float32))

G = tf.constant(np.array([3., 7., 10.])) #creating a tensor from a numpy array

np.array(G)  #converts back to numpy array

G.numpy() #also converts numpy array

#defaults of these are slightly different
numpy_G = tf.constant(np.array([3., 7., 10.])) #float64
tf_G = tf.constant([3., 7., 10.]) #float32