import torch
import torch.nn as nn
import numpy as np
import random
from seq2seq_dataset import ABSASeq2SeqDataset
from torch.utils.data import DataLoader

from apex import amp
from transformers import AdamW
from tqdm import tqdm
import logging
import math
from accelerate import Accelerator
from filelock import FileLock
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)


from tensorboardX import SummaryWriter
from utils import TensorboardAggregator
from sklearn.metrics import f1_score, accuracy_score
import os



MAX_LENGTH = 128


def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(model, config, accelerator, train_dataloader, eval_dataloader):
    logs_file = f'logs/{config["model"]}'
    writer = SummaryWriter(logs_file)
    agg = TensorboardAggregator(writer)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['lr'])


    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config['accum_steps'])
    max_train_steps = config['epochs'] * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps,
    )

    for epoch in tqdm(range(config['epochs'])):
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / config['accum_steps']
            accelerator.backward(loss)
            if step % config['accum_steps'] == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
    torch.save(model.state_dict(), 'model_with_sent.bin')
    model.eval()
    gen_kwargs = {
        "max_length": MAX_LENGTH,
        "num_beams": config['num_beams'],
    }
    for step, batch in enumerate(eval_dataloader):
        try:
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
                )
                generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)[0].replace('<pad>', '')
                decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=False)[0].replace('<pad>', '')
                print(decoded_preds)
                print(decoded_labels)
                print("================================")
        except:
            pass       
 
def eval(model, config, accelerator, eval_dataloader):
    model.eval()
    gen_kwargs = {
        "max_length": MAX_LENGTH,
        "num_beams": config['num_beams'],
    }
    all_preds = []
    all_labels = []
    all_texts = []
    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        input_texts = tokenizer.batch_decode(batch['input_ids'])
        for (i, text) in enumerate(input_texts):
          text  = text.replace("Opinion: ", "")
          input_texts[i] = text[:text.find("</s>")]
        all_texts += list(input_texts)
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]
            
            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)#[0].replace('<pad>', '')
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)#[0].replace('<pad>', '')
            all_preds += decoded_preds
            all_labels += decoded_labels
    all_preds = [x.replace("<pad>", "") for x in all_preds] 
    all_labels = [x.replace("<pad>", "") for x in all_labels]
    all_preds = [x.replace("</s>", "") for x in all_preds] 
    all_labels = [x.replace("</s>", "") for x in all_labels]
    import pandas as pd
    df = pd.DataFrame.from_dict({"text": all_texts, "label": all_labels, "prediction": all_preds})
    df.to_csv(f"result{data_test}.tsv", index=False, sep="\t")
    #fp = open(f"result{data_test}.txt", "w")
    #for (text, label, pred) in zip(input_texts, all_preds, all_labels):
    #    fp.write(text+ "\t" + label + "\t" + pred + "\n")

data_test = 'SP_REST_SB1_TEST.xml.gold'

if __name__ == '__main__':

    seed_torch()
    logging.basicConfig(level=logging.INFO)
    config = {
        'model': 'mt5',
        'lr': 5e-5,
        'epochs': 100,
        'batch_size': 4,
        'accum_steps': 2,
        'num_beams': None 
    }

    accelerator = Accelerator()
    logging.info(accelerator.state)
    #tokenizer = AutoTokenizer.from_pretrained("google/mt5-large")
    #model = AutoModelForSeq2SeqLM.from_pretrained('google/mt5-large')
    #torch.save(tokenizer, "t5large_tokenizer.pt")
    #torch.save(model, "t5large_seq2seqlm.pt")
    #exit()
    #tokenizer = torch.load("t5large-tokenizer.pt")
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("mt5-large")
    model = torch.load("t5large-seq2seqlm.pt")
    #model.load_state_dict(torch.load("model.bin"))
    #model.resize_token_embeddings(len(tokenizer) + 6)
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    logging.info("Prepare Train Dataset ...")
    train_dataset = ABSASeq2SeqDataset('data/train/ABSA16_Restaurants_Train_SB1_v2.xml', tokenizer)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=config['batch_size']
    )
    eval_dataset = ABSASeq2SeqDataset(f'data/test/{data_test}', tokenizer)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=config['batch_size'])
    if 1:
        model.load_state_dict(torch.load("model_with_sent.bin"))
        eval(model, config, accelerator, eval_dataloader)
    else:
        train(model, config, accelerator, train_dataloader, eval_dataloader)
