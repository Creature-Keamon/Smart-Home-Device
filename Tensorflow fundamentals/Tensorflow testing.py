
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import json
import numpy as np
import matplotlib.pyplot as plt

sentences = []
labels = []
vocab_size = 100000
embedding_dim = 20
max_length = 10000
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 28616

#reads json file
dataset = [json.loads(line) for line in open('Sarcasm_Headlines_Dataset_v2.json', 'r')]

#splits up the information in the json file and saves it in lists 
for item in dataset:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
 
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]


#defines tokenizer and tells it to add a '0' for every word there is different between sequences
tokenizer = Tokenizer(num_words = vocab_size, oov_token="<OOV>")

#updates internal knowledge of vocabulary
tokenizer.fit_on_texts(training_sentences)

#tokenises the text (assigns a numerical value to each)
word_index = tokenizer.word_index

#creates a sequence of numbers correlating to the words
training_sequences = tokenizer.texts_to_sequences(training_sentences)

#pads sequences with '0' to ensure that all sequences are equal length
training_padded = pad_sequences(training_sequences, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, padding=padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

num_epochs = 50
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)

sentence = ["I love spiders", "Grass is Green"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))