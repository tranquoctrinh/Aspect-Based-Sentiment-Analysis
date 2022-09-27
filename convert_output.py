import ast
import pandas as pd
import os
for file in os.listdir('output/'):
    if file.endswith("not_sentiment.csv"):
        df = pd.read_csv(f"output/{file}")
        df.target_true = df.target_true.apply(lambda x: ast.literal_eval(x))
        df.target_pred = df.target_pred.apply(lambda x: ast.literal_eval(x))
        df.target_true = df.target_true.apply(lambda x: ' '.join(x))
        df.target_pred = df.target_pred.apply(lambda x: ' '.join(x))
        df = df[['target_true', 'target_pred']]
        df.to_csv(f"{file}", header=False, index=False, sep="\t")
