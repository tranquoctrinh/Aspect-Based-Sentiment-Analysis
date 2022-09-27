import torch
import torch.nn as nn
import numpy as np
import random
from dataset import ABSADataset
from torch.utils.data import DataLoader

from apex import amp
from transformers import AdamW
from tqdm import tqdm
import logging
from model import ABSAModel

from tensorboardX import SummaryWriter
from utils import TensorboardAggregator
from sklearn.metrics import f1_score, accuracy_score
import os
from utils import TAG2IDX

def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(model, config, train_loader, val_loader):
    logs_file = f"logs/{config['model_base']}"
    writer = SummaryWriter(logs_file)
    agg = TensorboardAggregator(writer)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.05,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config["lr"]
    )

    model, optimizer = amp.initialize(
        model, optimizer, opt_level="O1", verbosity=0)

    fn_loss = nn.CrossEntropyLoss()

    for _ in range(config["epochs"]):
        model.train()
        for j, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
            token_ids, label = sample[0], sample[1]
            token_ids = token_ids.to(device)
            y_truth = label.long().to(device)
            y_pred = model(token_ids)
            if config['model_base'] == 'xlm-roberta-base':
                logits = y_pred
                attention_mask = token_ids != 1
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, len(TAG2IDX[config["sentiment"]]))
                active_labels = torch.where(
                    active_loss, y_truth.view(-1), torch.tensor(fn_loss.ignore_index).type_as(y_truth)
                )
                loss = fn_loss(active_logits, active_labels)
            else:
                loss = fn_loss(y_pred, y_truth)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if (j + 1) % config["accum_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()
            # y_pred = torch.softmax(y_pred, 1).argmax(1)
            # class_pred = [convert_label_to_tag(y.detach().cpu().tolist())[1] for y in y_pred]
            # accuracy = accuracy_score(class_pred, class_truth.detach().cpu().tolist())
            # , "train_accuracy": accuracy})
            agg.log({"train_loss": loss.item()})

        model.eval()
        result = []
        truth = []
        total_loss = []
        for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
            token_ids, label = sample[0], sample[1]
            token_ids = token_ids.to(device)
            y_truth = label.long().to(device)
            y_pred = model(token_ids)
            if config['model_base'] == 'xlm-roberta-base':
                logits = y_pred
                attention_mask = token_ids != 1
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, len(TAG2IDX[config["sentiment"]]))
                active_labels = torch.where(
                    active_loss, y_truth.view(-1), torch.tensor(fn_loss.ignore_index).type_as(y_truth)
                )
                loss = fn_loss(active_logits, active_labels)
            else:
                loss = fn_loss(y_pred, y_truth)
            loss = loss.detach().cpu()
            total_loss.append(loss.item())
        #   y_pred = torch.softmax(y_pred, 1).argmax(1)
        #     truth += list(class_truth.detach().cpu().tolist())
        #     class_pred = [convert_label_to_tag(y.detach().cpu().tolist())[1] for y in y_pred]
        #     result += class_pred
        # logging.info(f"Accuracy on Validation: {accuracy_score(truth, result)}")
        logging.info(
            f"Average Loss in Validation: {sum(total_loss) / len(total_loss)}")

    torch.save(model.state_dict(), config["model_path"])


if __name__ == "__main__":

    seed_torch()
    logging.basicConfig(level=logging.INFO)
    from utils import config
    logging.info("Prepare Train Dataset ...")
    train_dataset = ABSADataset(
        path="../data/train/ABSA16_Restaurants_Train_SB1_v2.xml",
        model_base=config["model_base"],
        sentiment=config["sentiment"]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_dataset = ABSADataset(
        path="../data/test/DU_REST_SB1_TEST.xml.gold",
        model_base=config["model_base"],
        sentiment=config["sentiment"]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )
    model = ABSAModel(
        model_base=config["model_base"],
        sentiment=config["sentiment"]
    )
    train(model, config, train_loader, val_loader)
