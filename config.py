from base import BaseConfigurator
from typing import Dict, Any, Optional


class ModelConfig(BaseConfigurator):
    """Configuration for model architecture."""
    
    def __init__(self, model_type: str = 'absa', **kwargs):
        """
        Initialize model configuration.
        
        Args:
            model_type: Type of model ('absa' or 'seq2seq')
            **kwargs: Additional configuration parameters
        """
        default_config = {
            'model_name': 'xlm-roberta-base',
            'dropout_rate': 0.3,
            'num_classes': 3,
        }
        
        if model_type == 'seq2seq':
            default_config.update({
                'model_name': 'mt5-base',
                'max_length': 128,
            })
        
        # Override defaults with provided kwargs
        default_config.update(kwargs)
        
        super(ModelConfig, self).__init__(default_config)


class TrainingConfig(BaseConfigurator):
    """Configuration for model training."""
    
    def __init__(self, model_type: str = 'absa', **kwargs):
        """
        Initialize training configuration.
        
        Args:
            model_type: Type of model ('absa' or 'seq2seq')
            **kwargs: Additional configuration parameters
        """
        default_config = {
            'lr': 5e-5,
            'epochs': 25,
            'batch_size': 16,
            'accum_steps': 2,
            'seed': 42,
            'opt_level': 'O1',
            'weight_decay': 0.05,
        }
        
        if model_type == 'seq2seq':
            default_config.update({
                'lr': 3e-5,
                'epochs': 30,
                'batch_size': 8,
            })
        
        # Override defaults with provided kwargs
        default_config.update(kwargs)
        
        super(TrainingConfig, self).__init__(default_config)


class DataConfig(BaseConfigurator):
    """Configuration for data processing."""
    
    def __init__(self, model_type: str = 'absa', **kwargs):
        """
        Initialize data configuration.
        
        Args:
            model_type: Type of model ('absa' or 'seq2seq')
            **kwargs: Additional configuration parameters
        """
        default_config = {
            'train_path': 'data/train/ABSA16_Restaurants_Train_SB1_v2.xml',
            'val_path': 'data/test/DU_REST_SB1_TEST.xml.gold',
            'max_length': 110,
            'special_tokens': {'additional_special_tokens': ['<e>', '</e>']},
        }
        
        if model_type == 'seq2seq':
            default_config.update({
                'max_length': 128,
                'special_tokens': {
                    'additional_special_tokens': ['<p>', '</p>', '<e>', '</e>', '<n>', '</n>']
                },
            })
        
        # Override defaults with provided kwargs
        default_config.update(kwargs)
        
        super(DataConfig, self).__init__(default_config)


class ProjectConfig:
    """Main configuration class for the project."""
    
    def __init__(self, model_type: str = 'absa', **kwargs):
        """
        Initialize project configuration.
        
        Args:
            model_type: Type of model ('absa' or 'seq2seq')
            **kwargs: Additional configuration parameters for sub-configs
        """
        self.model_type = model_type
        
        # Extract config-specific kwargs
        model_kwargs = kwargs.get('model', {})
        training_kwargs = kwargs.get('training', {})
        data_kwargs = kwargs.get('data', {})
        
        # Create sub-configurations
        self.model = ModelConfig(model_type, **model_kwargs)
        self.training = TrainingConfig(model_type, **training_kwargs)
        self.data = DataConfig(model_type, **data_kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            'model_type': self.model_type,
            'model': self.model.config,
            'training': self.training.config,
            'data': self.data.config,
        } 