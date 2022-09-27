from torch.utils.data import dataset
import xml.etree.ElementTree as ET
import torch
from transformers import AutoTokenizer, T5Tokenizer

SPECIAL_TOKENS = {'additional_special_tokens': ['<p>', '</p>', '<e>', '</e>', '<n>', '</n>']}
MAX_LENGTH = 128
CLASS_DICT = {'positive': 0, 'neutral': 1, 'negative': 2}


class ABSASeq2SeqDataset(dataset.Dataset):
    
    def __init__(self, path, tokenizer):

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
        self.inputs = []
        self.labels = []
        for sent_id in self.id_raw_data.keys():
            label = self.create_output_seq(self.id_raw_data[sent_id]['text'], self.id_raw_data[sent_id]['annotation'])
            if label == "<e>" or label == "<n>" or label == "<p>" or label == "":
                continue
            self.inputs.append(self.id_raw_data[sent_id]['text'])
            self.labels.append(label)
        self.tokenizer = tokenizer
#        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)

    def create_output_seq(self, label, ann):
        text = []
        sorted_ann = sorted(ann, key=lambda k: k['to'], reverse=True)
        pad = 0
        for op in ann:
            if int(op['from']) == 0 and int(op['to']) == 0:
                continue
                if op['polarity'] == 'negative':
                    text = ["<e>"]# </e>"#"<e> " + text + " </e>"
                elif op['polarity'] == 'neutral':
                    text = ["<n>"]# </n>"#"<n> " + text + " </n>"
                else:
                    text = ["<p>"]# </p>"#"<p> " + text + " </p>"
                break
                
            fromc = int(op['from'])# + pad
            toc = int(op['to'])# + pad
            s_add = " <e> "
            e_add = " </e> "
            #pad = pad + len(s_add) + len(e_add)
            if op['polarity'] == 'negative':
                text.append("negative " + label[fromc:toc])# + " </e>") #text[:fromc] + " <e> " + text[fromc:toc] + " </e> " + text[toc:]
                #text.append(label[fromc:toc])# + " </e>") #text[:fromc] + " <e> " + text[fromc:toc] + " </e> " + text[toc:]
            elif op['polarity'] == 'neutral':
                text.append("neutral " + label[fromc:toc])# + " </n>")#text[:fromc] + " <n> " + text[fromc:toc] + " </n> " + text[toc:]
                #text.append(label[fromc:toc])# + " </n>")#text[:fromc] + " <n> " + text[fromc:toc] + " </n> " + text[toc:]
            else:
                text.append("positive " + label[fromc:toc])# + " </p>")#text[:fromc] + " <p> " + text[fromc:toc] + " </p> " + text[toc:]
                #text.append(label[fromc:toc])# + " </p>")#text[:fromc] + " <p> " + text[fromc:toc] + " </p> " + text[toc:]
        return ' | '.join(text)

    def __getitem__(self, index):
        try:
            text = "Opinion: " + self.inputs[index]
        except:
            index = 0
            text = "Opinion: " + self.inputs[index]
        label = self.labels[index]
        model_input = self.tokenizer(text, max_length=MAX_LENGTH, padding='max_length', truncation=True)
        with self.tokenizer.as_target_tokenizer():
            model_label = self.tokenizer(label, max_length=MAX_LENGTH, padding='max_length', truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        model_label["input_ids"] =  [(l if l != self.tokenizer.pad_token_id else -100) for l in model_label["input_ids"]]
        model_input["labels"] = model_label["input_ids"]
        return model_input

    def __len__(self):
        return len(self.inputs)


    

if __name__=="__main__":
    tokenizer = T5Tokenizer.from_pretrained("mt5-large")
    data = ABSASeq2SeqDataset('data/train/ABSA16_Restaurants_Train_SB1_v2.xml', tokenizer)
    print(data[1])
    print(len(data))
    import ipdb; ipdb.set_trace()
