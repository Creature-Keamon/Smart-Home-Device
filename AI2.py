import tensorflow as tf
import csv
import json
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

label = []
data = []

import csv

import csv

def import_csv_file(file_path):
    data_lists = {}  # Dictionary to store the data as lists

    with open(file_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        headers = next(csvreader)  # Get the header row

        # Initialize lists for each column
        for header in headers:
            data_lists[header] = []

        # Append data to respective lists
        for row in csvreader:
            for idx, value in enumerate(row):
                data_lists[headers[idx]].append(value)

    # Convert dictionary values to separate lists
    column_lists = list(data_lists.values())

    return column_lists

if __name__ == "__main__":
    file_path = "lingSpam.csv"  # Replace with your CSV file path
    column_lists = import_csv_file(file_path)

    # Access data in separate lists by index
    label = column_lists[0]
    commas = column_lists[1]
    data = column_lists[2]

    # Print the separate lists
    print("Name List:", name_list)
    print("Age List:", age_list)
   

    

print (label)