# Aspect-Based Sentiment Analysis

An Object-Oriented implementation of Aspect-Based Sentiment Analysis (ABSA) using transformer models.

## Project Structure

The project has been refactored to follow Object-Oriented Programming principles:

- **Base Classes**:
  - `BaseModel`: Abstract base class for all models
  - `BaseDataset`: Abstract base class for all datasets
  - `BaseTrainer`: Abstract base class for all trainers
  - `BaseConfigurator`: Base class for configuration management

- **Models**:
  - `ABSAModel`: Transformer-based model for aspect sentiment classification

- **Datasets**:
  - `ABSADataset`: Dataset for aspect classification
  - `ABSASeq2SeqDataset`: Dataset for sequence-to-sequence generation

- **Configuration**:
  - `ProjectConfig`: Centralized configuration system
  - `ModelConfig`: Model-specific configuration
  - `TrainingConfig`: Training-specific configuration
  - `DataConfig`: Data-specific configuration

- **Training**:
  - `ABSATrainer`: Trainer for the ABSA model

## Usage

```bash
# Train ABSA model with default configuration
python main.py --model_type absa

# Train with custom parameters
python main.py --model_type absa --model_name xlm-roberta-base --batch_size 32 --lr 3e-5 --epochs 30

# Train Seq2Seq model
python main.py --model_type seq2seq --model_name mt5-base
```

## Requirements

- PyTorch
- Transformers
- Apex (for mixed precision training)
- scikit-learn
- TensorboardX
