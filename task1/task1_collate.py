import torch
import task1_dataset
from transformers import BertTokenizer, RobertaTokenizer, T5Tokenizer, ElectraTokenizer, XLNetTokenizer, CLIPTokenizer


class Collate:

    def __init__(self, class_list, is_unlabeled=False):

        '''
        class_list: path to the text file containing list of possible classes for task 1
        is_unlabeled: boolean, determines if the given dataset contains the gold labels 
        '''



        '''
        Based on the model used, the respective tokenizer should be employed.
        '''
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        # self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        # self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        # self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

        self.class_list = task1_dataset.read_classes(class_list)
        self.is_unlabeled = is_unlabeled

    
    def __call__(self, data):


        if len(data[0]) == 2:
            texts, labels = zip(*data)
            ids = None
        
        else:
            ids, texts, labels = zip(*data)

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

        if not self.is_unlabeled:
            for lo, c in zip(labels_output, labels):
                lo[c] = 1

        else:
            labels_output = None

       

        
        return ids, text_input, attention_masks, labels_output 





'''
The Collate function below should be used if the CLIP text encoder is used.
'''

# class Collate:
#     def __init__(self, class_list):
#         self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
#         self.class_list, _ = task1_dataset.read_classes(class_list)

#     def __call__(self, data):
#         ids, texts, labels = zip(*data)

#         # The CLIP tokenizer does not use 'encode_plus' method. It uses 'batch_encode_plus' or directly 'tokenizer' call.
#         tokenized_data = self.tokenizer(texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt", return_attention_mask=True)
        
#         input_ids = tokenized_data['input_ids']
#         attention_masks = tokenized_data['attention_mask']

#         labels_output = torch.zeros(len(texts), len(self.class_list))

#         # creating one-hot vector of labels for multi-label classification
#         for lo, c in zip(labels_output, labels):
#             lo[c] = 1

#         return ids, input_ids, attention_masks, labels_output
