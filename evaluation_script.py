# coding: utf-8
import pandas as pd
from collections import Counter
import string
import re
import argparse
import json
import sys
import ast
import os
import numpy as np
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

#files = [x for x in os.listdir() if x.startswith("result") and x.endswith("txt")]
files = [x for x in os.listdir() if x.endswith("no_sentiment.csv")]

for file in files:
	df = pd.read_csv(file)
	df = df[['target_true', 'target_pred']]
	df.columns = ['label', 'prediction']
	df.label = df.label.apply(lambda x: ' '.join(ast.literal_eval(x)))
	df.prediction = df.prediction.apply(lambda x: ' '.join(ast.literal_eval(x)))
	df.to_csv(file, sep="\t", index=False, header=False)


for file in files:
    df = pd.read_csv(file, sep="\t", header=None)
    print(len(df))
    df.columns = ['label', 'prediction']
    df.label = df.label.apply(lambda x: x.replace("</s>", ""))
    df.prediction = df.prediction.replace(np.nan, "")
    df.prediction = df.prediction.apply(lambda x: x.replace("</s>", ""))
    lst_f1 = [f1_score(x, y) for (x, y) in zip(df.label.values, df.prediction.values)]
    print(file)
    print("Score: ", sum(lst_f1) / len(lst_f1))
