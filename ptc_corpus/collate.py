import torch

import sys
sys.path.append('task1')
import task1_dataset

from transformers import BertTokenizer, RobertaTokenizer, T5Tokenizer, ElectraTokenizer, XLNetTokenizer




def read_classes(file_path):

    classes = []

    with open(file_path, 'r', encoding='utf8') as f:
    
        for label in f.readlines():
            label = label.strip()

            if label:
                classes.append(label)
        
    return classes




class Collate:


    def __init__(self):



        '''
        Based on the model used, the respective tokenizer should be employed.
        '''
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        # self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        # self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        # self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

        self.class_list = read_classes('ptc_corpus/classlist.txt')

    
    def __call__(self, data):

        # print(data)

        texts, labels = zip(*data)

        tokenized_text = []
        attention_masks = []

        for sent in texts:
            preprocessed_sent = sent.replace('\\n', ' ').strip()

            encoded_sent = self.tokenizer.encode_plus( 
                text=preprocessed_sent,
                add_special_tokens=True,
                max_length=512, 
                padding='max_length',
                truncation=True,
                return_attention_mask=True
            )

            tokenized_text.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))


        
        text_input = torch.tensor(tokenized_text)
        attention_masks = torch.tensor(attention_masks)

        texts_len = len(texts)
        labels_output = torch.zeros(texts_len, len(self.class_list))

        # print(texts_len, len(self.class_list))
        # print(labels_output.shape)
        # exit(0)


        for lo, c in zip(labels_output, labels):
            lo[c] = 1

        
        return text_input, attention_masks, labels_output 
