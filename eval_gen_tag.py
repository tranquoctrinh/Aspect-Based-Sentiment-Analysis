# coding: utf-8

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score


def tagging(index, span, tags):
    for i in range(len(span)):
        if i == 0:
            tags[index[0] + i] = 'B'
        else:
            tags[index[0] + i] = 'I'
    return tags
import pandas as pd
import ast
files = ['resultABSA16FR_Restaurants_Gold-withcontent.xml.tsv', 'resultDU_REST_SB1_TEST.xml.gold.tsv',  'resultEN_REST_SB1_TEST.xml.gold.tsv',  'resultSP_REST_SB1_TEST.xml.gold.tsv']
for file in files:
    print(file)
    df = pd.read_csv(file, sep = "\t")
    all_label_tags = []
    all_predict_tags = []
    for (text, labels, predicts) in zip(df.text.values, df.label.values, df.prediction.values):
        #text = text.replace("l'", "")
        text = text.split()
        label_tags = ['O'] * len(text)
        predict_tags = ['O'] * len(text)
        for label in labels.split('|'):
            label = label.strip().split()
            label_idx = [idx for idx in range(len(text) - len(label) + 1) if text[idx : idx + len(label)] == label]
            if len(label_idx) > 0:
                label_tags = tagging(label_idx, label, label_tags)
        for prediction in predicts.split('|'):
            prediction = prediction.strip().split()
            predict_idx = [idx for idx in range(len(text) - len(prediction) + 1) if text[idx : idx + len(prediction)] == prediction]
            if len(predict_idx) > 0:
                predict_tags = tagging(predict_idx, prediction, predict_tags)
        all_label_tags.append(label_tags)
        all_predict_tags.append(predict_tags)
    print(f"Accuracy: {accuracy_score(all_label_tags, all_predict_tags)}")
    print(f"F1-score: {f1_score(all_label_tags, all_predict_tags)}")

print("======================================================")

files = ['output/ABSA16FR_Restaurants_Gold-withcontent_xlm-roberta-base_no_sentiment..csv',  'output/EN_REST_SB1_TEST_xlm-roberta-base_no_sentiment..csv', 
'output/DU_REST_SB1_TEST_xlm-roberta-base_no_sentiment..csv'  ,'output/SP_REST_SB1_TEST_xlm-roberta-base_no_sentiment..csv']

#files = ['output/ABSA16FR_Restaurants_Gold-withcontent_google_mt5-base_no_sentiment..csv',   
#'output/DU_REST_SB1_TEST_google_mt5-base_no_sentiment..csv' , 'output/EN_REST_SB1_TEST_google_mt5-base_no_sentiment..csv','output/SP_REST_SB1_TEST_google_mt5-base_no_sentiment..csv']


for file in files:
    print(file)
    if file.endswith('csv'):
        df = pd.read_csv(file)
        df.columns = ['text', 'label', 'prediction']
        df['label'] = df.label.apply(lambda x: ast.literal_eval(x))
        df['prediction'] = df.prediction.apply(lambda x: ast.literal_eval(x))
    all_label_tags = []
    all_predict_tags = []
    for (text, labels, predicts) in zip(df.text.values, df.label.values, df.prediction.values):
        #text = text.replace("l'", "")
        text = text.split()
        label_tags = ['O'] * len(text)
        predict_tags = ['O'] * len(text)
        for label in labels:
            label = label.strip().split()
            label_idx = [idx for idx in range(len(text) - len(label) + 1) if text[idx : idx + len(label)] == label]
            if len(label_idx) > 0:
                label_tags = tagging(label_idx, label, label_tags)
        for prediction in predicts:
            prediction = prediction.strip().split()
            predict_idx = [idx for idx in range(len(text) - len(prediction) + 1) if text[idx : idx + len(prediction)] == prediction]
            if len(predict_idx) > 0:
                predict_tags = tagging(predict_idx, prediction, predict_tags)
        all_label_tags.append(label_tags)
        all_predict_tags.append(predict_tags)
    print(f"Accuracy: {accuracy_score(all_label_tags, all_predict_tags)}")
    print(f"F1-score: {f1_score(all_label_tags, all_predict_tags)}")
