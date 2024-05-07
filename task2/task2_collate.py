import torch
import task2_dataset
from transformers import BertTokenizer,  RobertaTokenizer, CLIPProcessor, CLIPTokenizer, AutoProcessor



class Collate:

    def __init__(self, class_list):

        '''
        class_list: path to the text file containing list of possible classes for task 1
        '''



        '''
        Based on the model used, the respective tokenizer should be employed.
        '''

        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        # self.processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base", apply_ocr=True)


        self.class_list = task2_dataset.read_classes(class_list)

    
    def __call__(self, data):


        if len(data[0]) == 3:
            texts, images, labels = zip(*data)  
            ids = None
        
        else:
            ids, texts, images, labels = zip(*data)

    
        tokenized_text = []
        attention_masks = []

        for sent in texts:
            preprocessed_sent = sent.replace('\\n', ' ').strip()

            encoded_sent = self.tokenizer.encode_plus(
                text=preprocessed_sent,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                return_attention_mask=True
            )

            tokenized_text.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        
        text_input = torch.tensor(tokenized_text)
        text_attention_masks = torch.tensor(attention_masks)

        texts_len = len(texts)
        labels_output = torch.zeros(texts_len, len(self.class_list))

        
        for lo, c in zip(labels_output, labels):
            lo[c] = 1


        '''
        If a CNN-based model is used, please comment out line 72 and use line 73 instead
        '''
        images_input = self.clip_processor(images=images, return_tensors="pt") 
        # images_input = torch.stack([image for image in images])

        return ids, text_input, text_attention_masks, images_input, labels_output







'''
The Collate function below should be used if the CLIP text encoder is used.
'''

# class Collate:

#     def __init__(self, class_list):

#         self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
#         self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#         self.class_list = task2_dataset.read_classes(class_list)

       
#     def __call__(self, data):

#         ids, texts, images, labels = zip(*data)
        
#         tokenized_data = self.tokenizer(texts, padding="max_length", truncation=True, 
#                                         max_length=77, return_tensors="pt", return_attention_mask=True)
        
#         input_ids = tokenized_data['input_ids']
#         attention_masks = tokenized_data['attention_mask']

#         labels_output = torch.zeros(len(texts), len(self.class_list))

        
#         for lo, c in zip(labels_output, labels):
#             lo[c] = 1


#         '''
#         If a CNN-based model is used, please comment out line 114 and use line 115 instead
#         '''
#         images_input = self.clip_processor(images=images, return_tensors="pt") #.pixel_values
#         # images_input = torch.stack([self.preprocess(image) for image in images])

#         return ids, input_ids, attention_masks, images_input, labels_output
