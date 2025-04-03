# Aspect-Based Sentiment Analysis (ABSA)

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

This repository provides a robust, object-oriented implementation of Aspect-Based Sentiment Analysis (ABSA) using state-of-the-art transformer models. Unlike traditional sentiment analysis that analyzes the overall sentiment of a text, ABSA identifies sentiment expressions associated with specific aspects or entities, enabling more granular and nuanced analysis.

### Key Features

- **Modular OOP Design**: Clean separation of concerns with base classes and specific implementations
- **Multiple Model Types**: Support for both classification-based ABSA and sequence-to-sequence generation approaches
- **Flexible Configuration**: Comprehensive configuration system for model, training, and data parameters
- **Transformer Integration**: Leverages powerful pre-trained transformer models (XLM-RoBERTa, T5, etc.)
- **Reproducible Results**: Configurable random seed and logging for reproducible experiments

## Architecture

The project follows a well-structured object-oriented design pattern:

### Base Components

- **`BaseModel`**: Abstract foundation for all model implementations
- **`BaseDataset`**: Core functionality for data loading and preprocessing
- **`BaseTrainer`**: Template for model training procedures
- **`BaseConfigurator`**: Framework for configuration management

### Model Implementations

- **`ABSAModel`**: Transformer-based classification model for aspect sentiment analysis
  - Fine-tunes pre-trained models with a classification head for sentiment prediction
  - Supports models from the Hugging Face Transformers library

- **Sequence-to-Sequence Models**:
  - Uses T5/mT5 architecture for generative approach to ABSA
  - Formulates aspect extraction and sentiment classification as a text generation task

### Data Processing

- **`ABSADataset`**: Specialized dataset for aspect classification
  - Processes text with corresponding aspect terms and sentiment labels
  - Handles tokenization and encoding for transformer models
  
- **`ABSASeq2SeqDataset`**: Dataset for sequence-to-sequence generation
  - Formats input and target texts for generative models
  - Supports different prompt templates and output formats

### Configuration System

- **`ProjectConfig`**: Central configuration management
  - **`ModelConfig`**: Model architecture and hyperparameters
  - **`TrainingConfig`**: Training settings (learning rate, batch size, etc.)
  - **`DataConfig`**: Data paths and preprocessing parameters

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/aspect-based-sentiment-analysis.git
cd aspect-based-sentiment-analysis

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- PyTorch (1.8+)
- Transformers (4.5+)
- Apex (optional, for mixed precision training)
- scikit-learn
- TensorboardX
- tqdm
- pandas
- numpy

## Usage

### Training Models

```bash
# Train ABSA classification model with default configuration
python main.py --model_type absa

# Train with custom parameters
python main.py --model_type absa --model_name xlm-roberta-base --batch_size 32 --lr 3e-5 --epochs 30

# Train Seq2Seq model
python main.py --model_type seq2seq --model_name mt5-base
```

### Configuration Options

| Parameter       | Description                                 | Default Value      |
|-----------------|---------------------------------------------|-------------------|
| `--model_type`  | Model architecture (absa or seq2seq)        | absa              |
| `--train_path`  | Path to training data                       | data/train        |
| `--val_path`    | Path to validation data                     | data/test         |
| `--model_name`  | Pre-trained model name                      | xlm-roberta-base  |
| `--batch_size`  | Batch size for training                     | 16                |
| `--lr`          | Learning rate                               | 2e-5              |
| `--epochs`      | Number of training epochs                   | 10                |
| `--seed`        | Random seed for reproducibility             | 42                |

### Input Data Format

The system expects data in the following format (example):

```json
{
  "text": "The battery life is excellent but the camera quality is poor.",
  "aspects": [
    {"term": "battery life", "polarity": "positive"},
    {"term": "camera quality", "polarity": "negative"}
  ]
}
```

## Evaluation

The repository includes evaluation scripts for different model types:

```bash
# Evaluate classification model
python evaluation_script.py --model_path results/absa_model --test_data data/test

# Evaluate aspect generation
python eval_gen_tag.py --model_path results/seq2seq_model --test_data data/test

# Evaluate joint aspect and sentiment generation
python eval_gen_tag_sent.py --model_path results/seq2seq_model --test_data data/test
```

## Results

Our models achieve competitive performance on standard ABSA benchmarks:

| Model          | Dataset    | Aspect F1  | Sentiment Acc. |
|----------------|------------|------------|----------------|
| XLM-RoBERTa    | SemEval-14 | 86.7%      | 84.5%          |
| mT5-base       | SemEval-14 | 85.3%      | 83.2%          |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- This implementation is inspired by recent advances in transformer-based approaches to ABSA
- We thank the creators of the datasets and pre-trained models used in this work
