from typing import Dict, Any, List, Optional
import pandas as pd
import ast
import os

class TensorboardAggregator:
    """
        Log average value periodically instead of logging on every batch
    """

    def __init__(self, writer, every: int = 5):
        """
        Initialize the aggregator.
        
        Args:
            writer: Tensorboard writer
            every: Frequency of logging
        """
        self.writer = writer
        self.every = every

        self.step = 0
        self.scalars = None

    def log(self, scalars: Dict[str, float]) -> None:
        """
        Log scalar values.
        
        Args:
            scalars: Dictionary of scalar values to log
        """
        self.step += 1

        if self.scalars is None:
            self.scalars = scalars.copy()
        else:
            for k, v in scalars.items():
                self.scalars[k] += v

        if self.step % self.every == 0:
            for k, v in self.scalars.items():
                self.writer.add_scalar(k, v / self.every, self.step)
            self.scalars = None

class OutputConverter:
    """
    Utility class for converting model outputs to standard formats.
    """
    
    @staticmethod
    def convert_single_file(file_path: str, output_path: Optional[str] = None) -> None:
        """
        Convert a single file with target_true and target_pred columns
        
        Args:
            file_path: Path to the input CSV file
            output_path: Path to save the output CSV file, if None uses the input path
        """
        if output_path is None:
            output_path = file_path
        
        df = pd.read_csv(file_path)
        df = df[['target_true', 'target_pred']]
        df.columns = ['label', 'prediction']
        df.label = df.label.apply(lambda x: ' '.join(ast.literal_eval(x)))
        df.prediction = df.prediction.apply(lambda x: ' '.join(ast.literal_eval(x)))
        df.to_csv(output_path, sep="\t", index=False, header=False)

    @staticmethod
    def convert_output_directory(output_dir: str = 'output/', pattern: str = 'not_sentiment.csv') -> None:
        """
        Convert all files in a directory matching a pattern
        
        Args:
            output_dir: Directory containing files to convert
            pattern: File suffix pattern to match
        """
        for file in os.listdir(output_dir):
            if file.endswith(pattern):
                file_path = f"{output_dir}/{file}"
                df = pd.read_csv(file_path)
                df.target_true = df.target_true.apply(lambda x: ast.literal_eval(x))
                df.target_pred = df.target_pred.apply(lambda x: ast.literal_eval(x))
                df.target_true = df.target_true.apply(lambda x: ' '.join(x))
                df.target_pred = df.target_pred.apply(lambda x: ' '.join(x))
                df = df[['target_true', 'target_pred']]
                df.to_csv(f"{file}", header=False, index=False, sep="\t")
