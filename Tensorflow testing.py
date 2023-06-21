#imports keras and some other tools
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

#list containing some basic text
sentences = [
	'I love my dog',
	'I love my cat',
	'Do you think my dog is amazing?'
	]

#defines tokenizer and tells it to add a '0' for every word there is different between sequences
tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")

#updates internal knowledge of vocabulary
tokenizer.fit_on_texts(sentences)

#tokenises the text (assigns a numerical value to each)
word_index = tokenizer.word_index

#creates a sequence of numbers correlating to the words
sequences = tokenizer.texts_to_sequences(sentences)

#pads sequences with '0' to ensure that all sequences are equal length
padded = pad_sequences(sequences, padding='post')

#prints everything
print(word_index)
print(sequences)
print(padded)
