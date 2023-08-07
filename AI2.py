import tensorflow as tf
import csv
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

labels = []
message = []
labels_int = []
ham = "ham"
spam = "spam"

#loads dataset
with open('spam.csv', 'r') as dataset:
    csv_reader = csv.reader(dataset)
    
    for line in csv_reader:
        labels.append(line[0])
        message.append(line[1])

        if ham in labels:
                labels_int.append(1)
        if spam in labels:
                labels_int.append(0)

message_list = list(message)

for x in range(len(labels_int)):
    print(labels_int[x], message_list[x])
