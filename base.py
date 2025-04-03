import abc
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional


class BaseDataset(Dataset, abc.ABC):
    """Abstract base class for all datasets in the project."""
    
    @abc.abstractmethod
    def __len__(self):
        """Return the number of samples in the dataset."""
        pass
    
    @abc.abstractmethod
    def __getitem__(self, index):
        """Return a sample from the dataset."""
        pass


class BaseModel(nn.Module, abc.ABC):
    """Abstract base class for all models in the project."""
    
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass of the model."""
        pass
    
    def save(self, path: str):
        """Save model weights to the specified path."""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        """Load model weights from the specified path."""
        self.load_state_dict(torch.load(path))


class BaseTrainer(abc.ABC):
    """Abstract base class for all trainers in the project."""
    
    @abc.abstractmethod
    def train(self, *args, **kwargs):
        """Train the model."""
        pass
    
    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        """Evaluate the model."""
        pass


class BaseConfigurator:
    """Base class for configuring model/training parameters."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def get(self, key: str, default=None):
        """Get a configuration parameter."""
        return self.config.get(key, default)
    
    def update(self, new_config: Dict[str, Any]):
        """Update configuration parameters."""
        self.config.update(new_config) 