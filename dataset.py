from torch.utils.data import dataset
import xml.etree.ElementTree as ET
import torch
from transformers import AutoTokenizer
from base import BaseDataset
from typing import List, Dict, Tuple, Optional, Any


# Constants
SPECIAL_TOKENS = {'additional_special_tokens': ['<e>', '</e>']}
MAX_LENGTH = 110
CLASS_DICT = {'positive': 0, 'neutral': 1, 'negative': 2}


class ABSADataset(BaseDataset):
    """
    Dataset for Aspect-Based Sentiment Analysis.
    """
    
    def __init__(self, path: str, model_name: str = 'xlm-roberta-base', max_length: int = MAX_LENGTH):
        """
        Initialize the ABSA dataset.
        
        Args:
            path: Path to the dataset XML file
            model_name: Name of the pretrained model for tokenization
            max_length: Maximum sequence length
        """
        self.path = path
        self.model_name = model_name
        self.max_length = max_length
        
        # Parse XML data
        self.id_raw_data = self._parse_xml()
        
        # Create samples
        self.all_samples = self._create_samples()
        
        # Add special characters for aspect marking
        self._add_special_character()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        self.start_id = self.tokenizer.all_special_ids[self.tokenizer.all_special_tokens.index('<e>')]
        self.end_id = self.tokenizer.all_special_ids[self.tokenizer.all_special_tokens.index('</e>')]

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
    
    def _create_samples(self) -> List[List]:
        """Create samples from the raw data."""
        all_samples = []
        for sent_id in self.id_raw_data.keys():
            for ann in self.id_raw_data[sent_id]['annotation']:
                sample = [
                    self.id_raw_data[sent_id]['text'], 
                    (int(ann['from']), int(ann['to'])), 
                    ann['polarity']
                ]
                all_samples.append(sample)
        return all_samples

    def _add_special_character(self) -> None:
        """Add special characters to mark aspects in the text."""
        for sample in self.all_samples:
            text = sample[0]
            fromc = sample[1][0]
            toc = sample[1][1]
            if fromc == 0 and toc == 0:  # general case 
                text = "<e> " + text + " </e>"
            else:
                text = text[:fromc] + " <e> " + text[fromc:toc] + " </e> " + text[toc:]
            sample[0] = text

    def convert_text_uncased(self, text: str) -> List[int]:
        """
        Convert text to token IDs with padding.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        if self.model_name == 'xlm-roberta-base':
            text_pad = "<s> " + text + " </s>"
            tokens_a = self.tokenizer.tokenize(text_pad)
            if len(tokens_a) > self.max_length:
                tokens_a = tokens_a[:self.max_length]
            one_token = self.tokenizer.convert_tokens_to_ids(tokens_a)
            one_token += [1] * (self.max_length - len(tokens_a))
            return one_token
        else:
            text_pad = "[CLS] " + text + " [SEP]"
            tokens_a = self.tokenizer.tokenize(text)[:self.max_length]
            if len(tokens_a) > self.max_length:
                tokens_a = tokens_a[:self.max_length]
            one_token = self.tokenizer.convert_tokens_to_ids(tokens_a)
            one_token += [0] * (self.max_length - len(tokens_a))
        return one_token
    
    def __getitem__(self, index: int) -> Tuple:
        """
        Get a sample from the dataset.
        
        Args:
            index: Index of the sample
            
        Returns:
            token_ids: Token IDs
            mask_opinion: Opinion mask
            label: Class label
            text: Original text
        """
        sample = self.all_samples[index]
        text = sample[0]
        token_ids = torch.LongTensor(self.convert_text_uncased(text))
        mask_opinion = (token_ids == self.start_id) + (token_ids == self.end_id)
        label = CLASS_DICT[sample[2]]
        return token_ids, mask_opinion, label, text

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.all_samples)


if __name__=="__main__":
    data = ABSADataset('data/train/ABSA16_Restaurants_Train_SB1_v2.xml')
    print(data[100])
    print(len(data))
    import ipdb; ipdb.set_trace()
