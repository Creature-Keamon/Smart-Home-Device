import tensorflow as tf

random_1 = tf.random.Generator.from_seed(42)

random_1 = random_1.normal(shape=(3, 2))

print(random_1)

random_2 = tf.random.Generator.from_seed(7)

random_2 = random_2.normal(shape=(3, 2))

print(random_2)