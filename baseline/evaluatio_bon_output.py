# coding: utf-8
import pandas as pd
import os
from transformers import AutoTokenizer
from tqdm import tqdm

from utils import *
from dataset import ABSADataset

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score


model_base = "google/mt5-base"
tokenizer = AutoTokenizer.from_pretrained(model_base)
tokenizer.add_special_tokens(SPECIAL_TOKENS)

start_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("<e>")]
end_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("</e>")]
end_sen_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index(
    "</s>")]
pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("<pad>")]
if model_base != "google/mt5-base":
    start_sen_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index(
        "<s>")]


def convert_text_uncased(text):
    text_pad = text
    tokens = tokenizer.tokenize(text_pad)
    if model_base != "google/mt5-base":
        ids = [start_sen_id] + \
            tokenizer.convert_tokens_to_ids(tokens) + [end_sen_id]
    else:
        ids = tokenizer.convert_tokens_to_ids(
            tokens) + [end_sen_id]

    if len(ids) > MAX_LENGTH:
        ids = ids[:MAX_LENGTH]
    else:
        ids += [pad_id] * (MAX_LENGTH - len(ids))

    return ids


def convert_text_to_tags(text):
    token_id = convert_text_uncased(text)
    label = []
    instring = 0
    idx_sen = -1
    for token in token_id:
        if token == start_id:
            instring = 1
            idx_sen += 1
            continue
        if token == end_id:
            instring = 0
            continue

        if instring == 0:
            label.append("O")
        elif instring == 1:
            instring += 1
            label.append("B")
        else:
            label.append("I")

    token_id = [token for token in token_id if token not in {start_id, end_id}]

    # padding
    if len(token_id) > MAX_LENGTH:
        label = label[:MAX_LENGTH]
    else:
        label += ["O"] * (MAX_LENGTH - len(label))
    return label


def main():
    files = [x for x in os.listdir('../') if x.endswith(".tsv")]
    for file in tqdm(files):
        data_file = "../data/test/" + \
            file.replace("result", "").replace(".tsv", "")
        labels, predictions = [], []
        label_tags, prediction_tags = [], []
        print("="*70)
        print(f"File: {file}")
        dataset = ABSADataset(data_file, model_base, False, "test")
        df = pd.read_csv('../' + file, sep="\t")
        for i, r in df.iterrows():
            text, label, prediction = r["text"], r["label"], r["prediction"]
            label_text = text
            for word in label.split("|"):
                word = word.strip()
                label_text = text.replace(word, f"<e> {word} </e>")

            pred_text = text
            for word in prediction.split("|"):
                word = word.strip()
                pred_text = text.replace(word, f"<e> {word} </e>")

            # labels.append(convert_text_uncased(label_text))
            # predictions.append(convert_text_uncased(pred_text))

            # label_tags.append(convert_text_to_tags(label_text))
            # prediction_tags.append(convert_text_to_tags(pred_text))

            # print("File Data Text: {}".format(dataset[i][-1]["text"]))
            # print(f"File Pred Text: {text}")
            # print("-"*100)

            label_tags.append([IDX2TAG[False][idx]
                              for idx in dataset[i][1].tolist()])
            prediction_tags.append(convert_text_to_tags(pred_text))

        print(f"Accuracy: {accuracy_score(label_tags, prediction_tags)}")
        print(f"F1-score: {f1_score(label_tags, prediction_tags)}")
        print("-"*20)
        print(classification_report(label_tags, prediction_tags))


if __name__ == "__main__":
    main()
