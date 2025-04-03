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
from base import BaseTrainer, BaseConfigurator
from typing import Dict, Any, Optional
from config import ProjectConfig


def seed_torch(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class ABSATrainer(BaseTrainer):
    """
    Trainer for Aspect-Based Sentiment Analysis models.
    """
    
    def __init__(self, model: ABSAModel, config: ProjectConfig, train_loader: DataLoader, val_loader: DataLoader):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup logging
        logs_file = f'logs/{config.model.get("model_name")}'
        self.writer = SummaryWriter(logs_file)
        self.agg = TensorboardAggregator(self.writer)
        
        # Setup device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Initialize mixed precision training
        self.model, self.optimizer = amp.initialize(
            self.model, 
            self.optimizer, 
            opt_level=config.training.get("opt_level"), 
            verbosity=0
        )

    def _create_optimizer(self) -> AdamW:
        """Create optimizer with weight decay."""
        no_decay = ["bias", "LayerNorm.weight"]
        weight_decay = self.config.training.get("weight_decay")
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.training.get('lr')
        )

    def train(self) -> None:
        """Train the model for the specified number of epochs."""
        for epoch in range(self.config.training.get('epochs')):
            logging.info(f"Starting epoch {epoch+1}/{self.config.training.get('epochs')}")
            self._train_epoch()
            self.evaluate()

    def _train_epoch(self) -> None:
        """Train the model for one epoch."""
        self.model.train()
        
        for j, sample in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            token_ids, mask_opinion, label = sample[0], sample[1], sample[2]
            token_ids = token_ids.to(self.device)
            mask_opinion = mask_opinion.to(self.device)
            y_truth = label.long().to(self.device)
            
            # Forward pass
            y_pred = self.model(token_ids, mask_opinion)
            loss = self.loss_fn(y_pred, y_truth)
            
            # Backward pass with mixed precision
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
                
            # Gradient accumulation
            if (j + 1) % self.config.training.get('accum_steps') == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            # Calculate accuracy and log metrics
            y_pred_classes = torch.softmax(y_pred, 1).argmax(1)
            accuracy = accuracy_score(y_pred_classes.detach().cpu().numpy(), y_truth.detach().cpu().numpy())
            self.agg.log({"train_loss": loss.item(), "train_accuracy": accuracy})

    def evaluate(self) -> float:
        """
        Evaluate the model on the validation set.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        result = []
        truth = []
        total_loss = []
        
        for j, sample in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
            token_ids, mask_opinion, label = sample[0], sample[1], sample[2]
            token_ids = token_ids.to(self.device)
            mask_opinion = mask_opinion.to(self.device)
            y_truth = label.long().to(self.device)
            
            # Forward pass
            with torch.no_grad():
                y_pred = self.model(token_ids, mask_opinion)
                loss = self.loss_fn(y_pred, y_truth).detach().cpu()
                
            total_loss.append(loss.item())
            y_pred_classes = torch.softmax(y_pred, 1).argmax(1)
            
            truth += list(y_truth.detach().cpu().numpy())
            result += list(y_pred_classes.detach().cpu().numpy())
            
        # Calculate and log metrics
        accuracy = accuracy_score(truth, result)
        avg_loss = sum(total_loss) / len(total_loss)
        
        logging.info(f'Accuracy on Validation: {accuracy}')
        logging.info(f'Average Loss in Validation: {avg_loss}')
        
        return avg_loss


def main() -> None:
    """Main function to run training."""
    # Create configuration
    config = ProjectConfig(model_type='absa')
    
    # Set random seed for reproducibility
    seed_torch(config.training.get('seed'))
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load datasets
    logging.info("Preparing Train Dataset...")
    train_dataset = ABSADataset(
        path=config.data.get('train_path'),
        model_name=config.model.get('model_name'),
        max_length=config.data.get('max_length')
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.get('batch_size'), 
        shuffle=True
    )
    
    logging.info("Preparing Validation Dataset...")
    val_dataset = ABSADataset(
        path=config.data.get('val_path'),
        model_name=config.model.get('model_name'),
        max_length=config.data.get('max_length')
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.training.get('batch_size'), 
        shuffle=False
    )
    
    # Create model
    model = ABSAModel(
        model_name=config.model.get('model_name'),
        dropout_rate=config.model.get('dropout_rate'),
        num_classes=config.model.get('num_classes')
    )
    
    # Create trainer and train
    trainer = ABSATrainer(model, config, train_loader, val_loader)
    trainer.train()


if __name__ == '__main__':
    main()




