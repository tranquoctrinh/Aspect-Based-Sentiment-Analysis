import torch
import torch.nn as nn
from transformers import AutoModel
from base import BaseModel


class ABSAModel(BaseModel):
    """
    Aspect-Based Sentiment Analysis model using transformer architecture.
    """

    def __init__(self, model_name='xlm-roberta-base', dropout_rate=0.3, num_classes=3):
        """
        Initialize the ABSA model.
        
        Args:
            model_name: Pretrained model name
            dropout_rate: Dropout probability
            num_classes: Number of classification classes
        """
        super(ABSAModel, self).__init__()

        self.model_name = model_name
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        
        self.bertmodel = AutoModel.from_pretrained(model_name)
        self.bertmodel.resize_token_embeddings(self.bertmodel.config.vocab_size + 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768*2, num_classes)

    def forward(self, input_ids, opinion_mask):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            opinion_mask: Mask for opinion tokens
            
        Returns:
            logits: Classification logits
        """
        # Handle padding token ID based on model type
        if self.model_name == 'xlm-roberta-base':
            attention_mask = input_ids != 1
        else:
            attention_mask = input_ids != 0
            
        output = self.bertmodel(input_ids, attention_mask=attention_mask)
        
        try:
            op_vec = output[0][opinion_mask].reshape((input_ids.shape[0], 2, output[0].shape[-1]))
        except:
            import ipdb; ipdb.set_trace()
            
        op_vec = op_vec.reshape(op_vec.shape[0], -1)
        op_vec = self.dropout(op_vec)
        logits = self.classifier(op_vec)
        
        return logits