import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import MT5EncoderModel, XLMRobertaModel, RobertaModel, AutoModelForTokenClassification
from utils import TAG2IDX


class ABSAModel(nn.Module):

    def __init__(self, model_base, sentiment):
        super(ABSAModel, self).__init__()
        self.model_base = model_base
        if self.model_base == "google/mt5-base":
            self.mt5model = MT5EncoderModel.from_pretrained(self.model_base)
            self.mt5model.resize_token_embeddings(
                self.mt5model.config.vocab_size + 2)
            self.classifier = nn.Linear(768, len(TAG2IDX[sentiment]))

        elif self.model_base == "xlm-roberta-base":
            self.model = AutoModel.from_pretrained(self.model_base)
            self.model.resize_token_embeddings(
                self.model.config.vocab_size + 2)
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(768, len(TAG2IDX[sentiment]))
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(
                'xlm-roberta-base')
            self.model.resize_token_embeddings(
                self.model.config.vocab_size + 2)
        print("DATASET model base:", self.model_base)
        print("DATASET sentiment:", sentiment)

    def forward(self, input_ids):
        if self.model_base == "google/mt5-base":
            outputs = self.mt5model(input_ids)
            output = outputs.last_hidden_state
            output = self.classifier(output)
            output = output.permute(0, 2, 1)
            return output

        elif self.model_base == "xlm-roberta-base":
            outputs = self.model(input_ids, attention_mask=input_ids != 1)
            sequence_output = outputs[0]
            sequence_output = self.dropout(sequence_output)
            output = self.classifier(sequence_output)
#            output = output.permute(0, 2, 1)
            return output
        else:
            outputs = self.model(**input_ids)
            return outputs
