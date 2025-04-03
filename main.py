import argparse
import logging
import torch
from torch.utils.data import DataLoader
from config import ProjectConfig
from dataset import ABSADataset
from seq2seq_dataset import ABSASeq2SeqDataset
from model import ABSAModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from train import ABSATrainer, seed_torch


def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("absa.log"),
            logging.StreamHandler()
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Aspect-Based Sentiment Analysis')
    
    parser.add_argument('--model_type', type=str, default='absa', choices=['absa', 'seq2seq'],
                      help='Model type to train (absa or seq2seq)')
    
    parser.add_argument('--train_path', type=str, 
                      help='Path to training data')
    
    parser.add_argument('--val_path', type=str, 
                      help='Path to validation data')
    
    parser.add_argument('--model_name', type=str, 
                      help='Pretrained model name')
    
    parser.add_argument('--batch_size', type=int, 
                      help='Batch size for training')
    
    parser.add_argument('--lr', type=float, 
                      help='Learning rate')
    
    parser.add_argument('--epochs', type=int, 
                      help='Number of training epochs')
    
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    return parser.parse_args()


def build_config(args):
    """Build configuration from command line arguments."""
    # Create config with specified model type
    config = ProjectConfig(model_type=args.model_type)
    
    # Update configuration with command line arguments
    if args.train_path:
        config.data.update({'train_path': args.train_path})
    
    if args.val_path:
        config.data.update({'val_path': args.val_path})
    
    if args.model_name:
        config.model.update({'model_name': args.model_name})
    
    if args.batch_size:
        config.training.update({'batch_size': args.batch_size})
    
    if args.lr:
        config.training.update({'lr': args.lr})
    
    if args.epochs:
        config.training.update({'epochs': args.epochs})
    
    if args.seed:
        config.training.update({'seed': args.seed})
    
    return config


def create_absa_model(config):
    """Create ABSA classifier model."""
    model = ABSAModel(
        model_name=config.model.get('model_name'),
        dropout_rate=config.model.get('dropout_rate'),
        num_classes=config.model.get('num_classes')
    )
    return model


def create_seq2seq_model(config):
    """Create Seq2Seq model."""
    model = T5ForConditionalGeneration.from_pretrained(config.model.get('model_name'))
    return model


def create_absa_dataloaders(config):
    """Create ABSA data loaders."""
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
    
    return train_loader, val_loader


def create_seq2seq_dataloaders(config):
    """Create Seq2Seq data loaders."""
    tokenizer = T5Tokenizer.from_pretrained(config.model.get('model_name'))
    
    train_dataset = ABSASeq2SeqDataset(
        path=config.data.get('train_path'),
        tokenizer=tokenizer,
        max_length=config.data.get('max_length')
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.get('batch_size'), 
        shuffle=True
    )
    
    val_dataset = ABSASeq2SeqDataset(
        path=config.data.get('val_path'),
        tokenizer=tokenizer,
        max_length=config.data.get('max_length')
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.training.get('batch_size'), 
        shuffle=False
    )
    
    return train_loader, val_loader


def main():
    """Main entry point of the application."""
    # Setup logging
    setup_logging()
    
    # Parse command line arguments
    args = parse_args()
    
    # Build configuration
    config = build_config(args)
    
    # Set random seed for reproducibility
    seed_torch(config.training.get('seed'))
    
    # Create model and data loaders based on model type
    if config.model_type == 'absa':
        logging.info("Creating ABSA model and data loaders...")
        model = create_absa_model(config)
        train_loader, val_loader = create_absa_dataloaders(config)
        
        # Create trainer and train
        trainer = ABSATrainer(model, config, train_loader, val_loader)
        trainer.train()
        
    elif config.model_type == 'seq2seq':
        logging.info("Creating Seq2Seq model and data loaders...")
        model = create_seq2seq_model(config)
        train_loader, val_loader = create_seq2seq_dataloaders(config)
        
        # Create trainer and train (would need a Seq2SeqTrainer implementation)
        logging.info("Seq2Seq training not yet implemented")
    
    else:
        logging.error(f"Unsupported model type: {config.model_type}")


if __name__ == '__main__':
    main() 