# coding: utf-8
import pandas as pd
import ast
df = pd.read_csv("EN_REST_SB1_TEST_XML-RoBERTa_pred_no_sentiment.csv")
df = df[['target_true', 'target_pred']]
df.columns = ['label', 'prediction']
df.label = df.label.apply(lambda x: ' '.join(ast.literal_eval(x)))
df.prediction = df.prediction.apply(lambda x: ' '.join(ast.literal_eval(x)))
df.to_csv("EN_REST_SB1_TEST_XML-RoBERTa_pred_no_sentiment.csv", sep="\t", index=False, header=False)
