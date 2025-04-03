from torch.utils.data import dataset
import xml.etree.ElementTree as ET
import torch
from transformers import AutoTokenizer, T5Tokenizer
from base import BaseDataset
from typing import Dict, List, Tuple, Any, Optional

SPECIAL_TOKENS = {'additional_special_tokens': ['<p>', '</p>', '<e>', '</e>', '<n>', '</n>']}
MAX_LENGTH = 128
CLASS_DICT = {'positive': 0, 'neutral': 1, 'negative': 2}


class ABSASeq2SeqDataset(BaseDataset):
    """
    Sequence-to-Sequence Dataset for Aspect-Based Sentiment Analysis.
    """
    
    def __init__(self, path: str, tokenizer, max_length: int = MAX_LENGTH):
        """
        Initialize the ABSA Seq2Seq dataset.
        
        Args:
            path: Path to the dataset XML file
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
        """
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Parse XML data
        self.id_raw_data = self._parse_xml()
        
        # Create input and label sequences
        self.inputs, self.labels = self._create_sequences()

    def _parse_xml(self) -> Dict[str, Dict]:
        """Parse XML file and extract raw data."""
        tree = ET.parse(self.path)
        root = tree.getroot()
        all_raw_sentences = root.findall('**/sentence')
        
        id_raw_data = {}
        for sentence in all_raw_sentences:
            id = sentence.attrib['id']
            id_raw_data[id] = {}
            id_raw_data[id]['text'] = sentence[0].text
            id_raw_data[id]['annotation'] = []
            if len(sentence) >= 2:
                for ann in sentence[1]:
                    id_raw_data[id]['annotation'].append(ann.attrib)
        
        return id_raw_data
    
    def _create_sequences(self) -> Tuple[List[str], List[str]]:
        """Create input and label sequences from the raw data."""
        inputs = []
        labels = []
        
        for sent_id in self.id_raw_data.keys():
            label = self.create_output_seq(
                self.id_raw_data[sent_id]['text'], 
                self.id_raw_data[sent_id]['annotation']
            )
            
            if label == "<e>" or label == "<n>" or label == "<p>" or label == "":
                continue
                
            inputs.append(self.id_raw_data[sent_id]['text'])
            labels.append(label)
            
        return inputs, labels

    def create_output_seq(self, label: str, ann: List[Dict]) -> str:
        """
        Create output sequence for the given annotation.
        
        Args:
            label: Original text
            ann: List of annotations
            
        Returns:
            Formatted output sequence
        """
        text = []
        sorted_ann = sorted(ann, key=lambda k: k['to'], reverse=True)
        pad = 0
        
        for op in ann:
            if int(op['from']) == 0 and int(op['to']) == 0:
                continue
                if op['polarity'] == 'negative':
                    text = ["<e>"]
                elif op['polarity'] == 'neutral':
                    text = ["<n>"]
                else:
                    text = ["<p>"]
                break
                
            fromc = int(op['from'])
            toc = int(op['to'])
            
            if op['polarity'] == 'negative':
                text.append("negative " + label[fromc:toc])
            elif op['polarity'] == 'neutral':
                text.append("neutral " + label[fromc:toc])
            else:
                text.append("positive " + label[fromc:toc])
                
        return ' | '.join(text)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            index: Index of the sample
            
        Returns:
            Dictionary containing model inputs and labels
        """
        try:
            text = "Opinion: " + self.inputs[index]
        except:
            index = 0
            text = "Opinion: " + self.inputs[index]
            
        label = self.labels[index]
        
        # Tokenize input
        model_input = self.tokenizer(
            text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True
        )
        
        # Tokenize label
        with self.tokenizer.as_target_tokenizer():
            model_label = self.tokenizer(
                label, 
                max_length=self.max_length, 
                padding='max_length', 
                truncation=True
            )
            
        # Replace padding tokens with -100 to ignore in loss calculation
        model_label["input_ids"] = [
            (l if l != self.tokenizer.pad_token_id else -100) 
            for l in model_label["input_ids"]
        ]
        
        model_input["labels"] = model_label["input_ids"]
        return model_input

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.inputs)


if __name__=="__main__":
    tokenizer = T5Tokenizer.from_pretrained("mt5-large")
    data = ABSASeq2SeqDataset('data/train/ABSA16_Restaurants_Train_SB1_v2.xml', tokenizer)
    print(data[1])
    print(len(data))
    import ipdb; ipdb.set_trace()
