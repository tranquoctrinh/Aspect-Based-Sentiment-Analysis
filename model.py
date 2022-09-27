import torch
import torch.nn as nn
from transformers import AutoModel

class ABSAModel(nn.Module):

    def __init__(self):
        super(ABSAModel, self).__init__()

        self.bertmodel = AutoModel.from_pretrained('xlm-roberta-base')
        self.bertmodel.resize_token_embeddings(self.bertmodel.config.vocab_size + 2)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768*2 , 3)

    def forward(self, input_ids, opinion_mask):
        if self.bertmodel.name_or_path == 'xlm-roberta-base':
            output = self.bertmodel(input_ids, attention_mask=input_ids != 1)
        else:
            output = self.bertmodel(input_ids, attention_mask=input_ids != 0)
        try:
            op_vec = output[0][opinion_mask].reshape((input_ids.shape[0], 2, output[0].shape[-1]))
        except:
            import ipdb; ipdb.set_trace()
        op_vec = op_vec.reshape(op_vec.shape[0], -1)
        op_vec = self.dropout(op_vec)
        logits = self.classifier(op_vec)
        return logits