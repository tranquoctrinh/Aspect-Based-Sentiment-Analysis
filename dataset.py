from torch.utils.data import dataset
import xml.etree.ElementTree as ET
import torch
from transformers import AutoTokenizer

SPECIAL_TOKENS = {'additional_special_tokens': ['<e>', '</e>']}
MAX_LENGTH = 110
CLASS_DICT = {'positive': 0, 'neutral': 1, 'negative': 2}


class ABSADataset(dataset.Dataset):
    
    def __init__(self, path, type='train'):

        self.path = path
        tree = ET.parse(self.path)
        root = tree.getroot()
        all_raw_sentences = root.findall('**/sentence')
        self.id_raw_data = {}
        for sentence in all_raw_sentences:
            id = sentence.attrib['id']
            self.id_raw_data[id] = {}
            self.id_raw_data[id]['text'] = sentence[0].text
            self.id_raw_data[id]['annotation'] = []
            if len(sentence) >= 2:
                for ann in sentence[1]:
                    self.id_raw_data[id]['annotation'].append(ann.attrib)
        self.all_samples = []
        for sent_id in self.id_raw_data.keys():
            for ann in self.id_raw_data[sent_id]['annotation']:
                sample = [self.id_raw_data[sent_id]['text'], (int(ann['from']), int(ann['to'])), ann['polarity']]
                self.all_samples.append(sample)

        self.add_special_character()
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        self.start_id = self.tokenizer.all_special_ids[self.tokenizer.all_special_tokens.index('<e>')]
        self.end_id = self.tokenizer.all_special_ids[self.tokenizer.all_special_tokens.index('</e>')]


    def add_special_character(self):
        for sample in self.all_samples:
            text = sample[0]
            fromc = sample[1][0]
            toc = sample[1][1]
            if fromc == 0 and toc == 0: #general case 
                text = "<e> " + text + " </e>"
            else:
                text = text[:fromc] + " <e> " + text[fromc:toc] + " </e> " + text[toc:]
            sample[0] = text

    def convert_text_uncased(self, text):
        if self.tokenizer.name_or_path == 'xlm-roberta-base':
            text_pad = "<s> " + text + " </s>"
            tokens_a = self.tokenizer.tokenize(text_pad)
            if len(tokens_a) > MAX_LENGTH:
                tokens_a = tokens_a[:MAX_LENGTH]
            one_token = self.tokenizer.convert_tokens_to_ids(tokens_a)
            one_token += [1] * (MAX_LENGTH - len(tokens_a))
            return one_token
        else:
            text_pad = "[CLS] " + text + " [SEP]"
            tokens_a = self.tokenizer.tokenize(text)[:MAX_LENGTH]
            if len(tokens_a) > MAX_LENGTH:
                tokens_a = tokens_a[:MAX_LENGTH]
            one_token = self.tokenizer.convert_tokens_to_ids(tokens_a)
            one_token += [0] * (MAX_LENGTH - len(tokens_a))
        return one_token
        #return torch.LongTensor(one_token)
    
    def __getitem__(self, index):
        sample = self.all_samples[index]
        text = sample[0]
        token_ids = torch.LongTensor(self.convert_text_uncased(text))
        mask_opinion = (token_ids == self.start_id) + (token_ids == self.end_id)
        label = CLASS_DICT[sample[2]]
        return token_ids, mask_opinion, label, text

    def __len__(self):
        return len(self.all_samples)


    

if __name__=="__main__":
    data = ABSADataset('data/train/ABSA16_Restaurants_Train_SB1_v2.xml')
    print(data[100])
    print(len(data))
    import ipdb; ipdb.set_trace()
