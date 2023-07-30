import tensorflow as tf

not_shuffled = tf.constant([[10, 7], [3, 4], [2, 5]])

not_shuffled.ndim


tf.random.shuffle(not_shuffled, seed=42)

