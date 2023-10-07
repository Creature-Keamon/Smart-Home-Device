import random
import csv
import pandas as pd

labels_str = []
message = []
labels_int = []
 
# original lists
with open('spam.csv', 'r') as dataset:
    csv_reader = csv.reader(dataset)
    
    #splits dataset into two lists
    for line in csv_reader:
        labels_str.append(line[0])
        message.append(line[1])
    
    #creates a new list and adds 1s and 0s to it based on "ham"s and "spam"s in labels_str respectively
    for line in labels_str:
        if "ham" in line:
                labels_int.append(0)
        elif "spam" in line:
                labels_int.append(1)
                
with open('emails.csv', 'r') as dataset2:
    csv_reader2 = csv.reader(dataset2)
    
    for line in csv_reader2:
        labels_int.append(line[1])
        message.append(line[0])
 
# shuffle the lists with same order
zipped = list(zip(message, labels_int))
random.shuffle(zipped)
list1, list2 = zip(*zipped)

#turns lists into a dictionary
dict = {'message': list1, 'label': list2}

#turn dictionary into a pandas dataframe
df = pd.DataFrame(dict)

#turns dataframe into a csv
df.to_csv('full_spam_set.csv', index=False)
