from torch.utils.data import dataset
import xml.etree.ElementTree as ET
import torch
from transformers import AutoTokenizer
from collections import Counter

from utils import *


class ABSADataset(dataset.Dataset):

    def __init__(self, path, model_base, sentiment, phase="train"):

        self.path = path
        self.model_base = model_base
        self.sentiment = sentiment
        self.phase = phase
        tree = ET.parse(self.path)
        root = tree.getroot()
        all_raw_sentences = root.findall("**/sentence")
        self.id_raw_data = {}
        for sentence in all_raw_sentences:
            id = sentence.attrib["id"]
            self.id_raw_data[id] = {}
            self.id_raw_data[id]["text"] = sentence[0].text
            self.id_raw_data[id]["annotation"] = []
            if len(sentence) >= 2:
                for ann in sentence[1]:
                    self.id_raw_data[id]["annotation"].append(ann.attrib)
        self.all_samples = []
        for sent_id in self.id_raw_data.keys():
            sample = {
                "text": self.id_raw_data[sent_id]["text"], "idx": [], "opinion": []}
            for ann in self.id_raw_data[sent_id]["annotation"]:
                idx = (int(ann["from"]), int(ann["to"]))
                if sum(idx) != 0 and idx not in sample["idx"]:
                    sample["idx"].append(idx)
                    sample["opinion"].append(ann["polarity"])
            if len(sample["idx"]) != 0:
                self.all_samples.append(sample)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_base)  # "google/mt5-base")
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)

        self.start_id = self.tokenizer.all_special_ids[self.tokenizer.all_special_tokens.index(
            "<e>")]
        self.end_id = self.tokenizer.all_special_ids[self.tokenizer.all_special_tokens.index(
            "</e>")]
        self.end_sen_id = self.tokenizer.all_special_ids[self.tokenizer.all_special_tokens.index(
            "</s>")]
        self.pad_id = self.tokenizer.all_special_ids[self.tokenizer.all_special_tokens.index(
            "<pad>")]
        if self.model_base != "google/mt5-base":
            self.start_sen_id = self.tokenizer.all_special_ids[self.tokenizer.all_special_tokens.index(
                "<s>")]
        print("DATASET model base:", self.model_base)
        print("DATASET sentiment:", self.sentiment)

    def convert_text_uncased(self, text):
        text_pad = text
        tokens = self.tokenizer.tokenize(text_pad)
        if self.model_base != "google/mt5-base":
            ids = [self.start_sen_id] + \
                self.tokenizer.convert_tokens_to_ids(
                    tokens) + [self.end_sen_id]
        else:
            ids = self.tokenizer.convert_tokens_to_ids(
                tokens) + [self.end_sen_id]
        return ids
        # return torch.LongTensor(one_token)

    def add_special_character(self, sample):
        fromto = [(idx[0], idx[1], opinion)
                  for idx, opinion in zip(sample["idx"], sample["opinion"])]
        text = sample["text"]
        fromto = sorted(fromto, key=lambda item: item[0])
        target = []
        for i, idx in enumerate(fromto[::-1]):
            fromc = idx[0]
            toc = idx[1]
            target.append(text[fromc:toc])
            text = text[:fromc] + " <e> " + \
                text[fromc:toc] + " </e> " + text[toc:]

        new_sample = dict(
            text=' '.join(text.split()),
            idx=[(idx[0], idx[1]) for idx in fromto],
            opinion=[idx[2] for idx in fromto],
            target=[tg for tg in target[::-1]]
        )
        return new_sample

    def convert_uncased_label(self, sample):
        new_sample = self.add_special_character(sample)
        token_id = self.convert_text_uncased(new_sample["text"])
        opinion = new_sample["opinion"]

        label = []
        instring = 0
        idx_sen = -1
        for token in token_id:
            if token == self.start_id:
                instring = 1
                idx_sen += 1
                continue
            if token == self.end_id:
                instring = 0
                continue

            if instring == 0:
                label.append(TAG2IDX[self.sentiment]["O"])
            elif instring == 1:
                instring += 1
                if self.sentiment:
                    label.append(TAG2IDX[self.sentiment]
                                 [f"B-{opinion[idx_sen]}"])
                else:
                    label.append(TAG2IDX[self.sentiment]["B"])
            else:
                if self.sentiment:
                    label.append(TAG2IDX[self.sentiment]
                                 [f"I-{opinion[idx_sen]}"])
                else:
                    label.append(TAG2IDX[self.sentiment]["I"])
        token_id = [token for token in token_id if token not in {
            self.start_id, self.end_id}]

        # padding
        if len(token_id) > MAX_LENGTH:
            token_id = token_id[:MAX_LENGTH]
            label = label[:MAX_LENGTH]
        else:
            token_id += [self.pad_id] * (MAX_LENGTH - len(token_id))
            label += [TAG2IDX[self.sentiment]["O"]] * (MAX_LENGTH - len(label))

        new_sample["text"] = sample["text"]
        return token_id, label, new_sample

    def __getitem__(self, index):
        sample = self.all_samples[index]
        token_ids, label, new_sample = self.convert_uncased_label(sample)
        token_ids = torch.LongTensor(token_ids)
        label = torch.LongTensor(label)

        if self.phase in {"test"}:
            return token_ids, label, new_sample
        else:
            return token_ids, label

    def __len__(self):
        return len(self.all_samples)


if __name__ == "__main__":
    data = ABSADataset("../data/train/ABSA16_Restaurants_Train_SB1_v2.xml")
    print(data[100])
    print(len(data))
    import ipdb
    ipdb.set_trace()
