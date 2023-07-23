import tensorflow as  tf 

tf.Variable

changeable_tensor = tf.Variable([10, 7])
unchangeable_tensor = tf.constant([10, 7])

changeable_tensor[0].assign(7)

unchangeable_tensor[0].assign(7)
