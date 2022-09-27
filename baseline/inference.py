import torch
from dataset import ABSADataset

from tqdm import tqdm
import logging
from model import ABSAModel
import os

from utils import IDX2TAG, config
import pandas as pd

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score


def convert_tag_to_idx(tags):
    idx = []
    opinion = []
    instring = 0
    start, end = -1, -1
    for i, tag in enumerate(tags):
        if tag in {"B", "B-positive", "B-negative"}:
            instring = 1
            start = i
            if tag == "B-positive":
                opinion.append("positive")
            if tag == "B-negative":
                opinion.append("negative")
        elif tag in {"I", "I-positive", "I-negative"}:
            if instring:
                end = i + 1
        elif tag in {"O"}:
            instring = 0
            if start != -1 and end != -1:
                idx.append((start, end))
                start, end = -1, -1
    return idx, opinion


def inference(model, config, path_df):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_dataset = ABSADataset(
        path=path_df,
        model_base=config["model_base"],
        sentiment=config["sentiment"],
        phase="test"
    )

    test_result = []
    all_tags_label, all_tags_pred = [], []
    for sample in tqdm(test_dataset):
        token_ids, label, info = sample[0], sample[1], sample[2]
        token_ids = token_ids.unsqueeze(0).to(device)

        y_pred = model(token_ids)
        if config['model_base'] == 'xlm-roberta-base':
            y_pred = y_pred.permute(0, 2, 1)
        y_pred = torch.softmax(y_pred, 1).argmax(1)[0].tolist()

        tags_label = [IDX2TAG[config["sentiment"]][idx]
                      for idx in label.tolist()]
        tags_pred = [IDX2TAG[config["sentiment"]][idx] for idx in y_pred]

        all_tags_label.append(tags_label)
        all_tags_pred.append(tags_pred)

        idx, opinion = convert_tag_to_idx(tags_pred)
        target_pred = []
        for start, end in idx:
            token_pred = test_dataset.tokenizer.convert_ids_to_tokens(
                token_ids[0][start:end].detach().cpu().tolist())
            string_target_pred = test_dataset.tokenizer.convert_tokens_to_string(
                token_pred).strip()
            target_pred.append(string_target_pred)
        if config["sentiment"]:
            result = dict(
                text=info["text"],
                target_true=[(tg, op) for tg, op in zip(
                    info["target"], info["opinion"])],
                target_pred=[(tg, op) for tg, op in zip(target_pred, opinion)],
            )
        else:
            result = dict(
                text=info["text"],
                target_true=info["target"],
                target_pred=target_pred
            )
        test_result.append(result)
    print(f"Accuracy: {accuracy_score(all_tags_label, all_tags_pred)}")
    print(f"F1-score: {f1_score(all_tags_label, all_tags_pred)}")
    print("-"*20)
    print(classification_report(all_tags_label, all_tags_pred))
    return test_result


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Inference Test Dataset ...")

    model = ABSAModel(
        model_base=config["model_base"],
        sentiment=config["sentiment"]
    )
    print(config["model_path"])
    model.load_state_dict(torch.load(config["model_path"]))

    test_files = os.listdir("../data/test/")
    for i, file in enumerate(test_files):
        print("="*70)
        print(f"File [{i+1}/{len(test_files)}]: {file}")
        path_df = os.path.join("../data/test/", file)
        test_result = inference(model, config, path_df)
        df_result = pd.DataFrame(test_result)

        file_result_path = "../output/"
        file_result_path += file.split(".")[0]
        file_result_path += config["model_path"].replace(
            "model_seq_labeling", "").replace(".pt", ".csv")

        df_result.to_csv(file_result_path, index=False)


if __name__ == "__main__":
    main()
