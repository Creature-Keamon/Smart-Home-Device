import tensorflow as tf
import json
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

verdict = []
speaker = []
statement = []
date = []
source = []
fact_checker = []
check_date = []
analysis_link = []
verdict_int = []

#loads json file
dataset = [json.loads(line) for line in open('Sarcasm_Headlines_Dataset_v2.json', 'r')]

#splits up the information in the json file and saves it in lists 
for item in dataset:
    verdict.append(item['verdict'])
    speaker.append(item['statement_originator'])
    statement.append(item['statement'])
    date.append(item['statement_date'])
    source.append(item['statement_source'])
    fact_checker.append(item['factchecker'])
    check_date.append(item['factcheck_date'])
    analysis_link.append(item['factcheck_analysis_link'])

    #adds integers to a list based on verdict
    for item in verdict:
        if item == "false":
            verdict_int.append(item[1])
        elif item == "pants-fire":
            verdict_int.append(item[0])
        elif item == "mostly-false":
            verdict_int.append(item[2])
        elif item == "half-true":
            verdict_int.append(item[3])
        elif item == "mostly-true":
            verdict_int.append(item[4])
        elif item =="true":
            verdict_int.append(item[5])

training_verdicts = verdict_int[:16922]
testing_verdicts = verdict_int[16922:]