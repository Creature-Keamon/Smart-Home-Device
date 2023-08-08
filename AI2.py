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

        for ham in line:
                labels_int.append(1)
        for spam in line:
                labels_int.append(0)

message_list = list(message)

print(labels_int[3],",", message_list[3])
