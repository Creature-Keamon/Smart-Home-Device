from pyexpat import model
import tensorflow as tf
from tensorflow import keras

#load in the model
model_reload = tf.keras.models.load_model("spam_detection_model")

#show summary of model
model_reload.summary()