import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, RobertaConfig, RobertaModel, T5EncoderModel, T5Config, CLIPModel, ElectraModel, ElectraConfig, XLNetModel, XLNetConfig


class TextEncoderBERT(nn.Module):
    
    def __init__(self):
        super(TextEncoderBERT, self).__init__()

        self.bert_config = BertConfig.from_pretrained('bert-base-cased', 
                                                 output_hidden_states=True, 
                                                 num_hidden_layers=10)
        
        self.model = BertModel.from_pretrained('bert-base-cased', config=self.bert_config)

        # self.classifier = nn.Linear(self.bert_config.hidden_size, 14)

        self.classifier = nn.Sequential(
            nn.Linear(self.bert_config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 20)  # If pretraining with PTC, change to 14
        )

    
    
    def forward(self, text_inputs, attention_mask):

        outputs = self.model(text_inputs, attention_mask=attention_mask)
        sequence_output = outputs[0][:,0,:] 

        logits = self.classifier(sequence_output)  
        
        return logits
    


class TextEncoderRoBERTa(nn.Module):

    def __init__(self):
        super(TextEncoderRoBERTa, self).__init__()

        self.roberta_config = RobertaConfig.from_pretrained('roberta-base', 
                                                            output_hidden_states=True, 
                                                            num_hidden_layers=10)
        
        self.model = RobertaModel.from_pretrained('roberta-base', config=self.roberta_config)

        # self.classifier = nn.Linear(self.roberta_config.hidden_size, 20)

        self.classifier = nn.Sequential(
            nn.Linear(self.roberta_config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 20) # If pretraining with PTC, change to 14
        )

    
    def forward(self, input_ids, attention_mask):
       
        outputs = self.model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0][:,0,:]  
        
        logits = self.classifier(sequence_output)
        
        return logits
    



class TextEncoderT5(nn.Module):

    def __init__(self, num_labels=20): # If pretraining with PTC, change num_labels to 14
        super(TextEncoderT5, self).__init__()

        self.t5_config = T5Config.from_pretrained('t5-base', 
                                                  return_dict=True)
        
       
        self.model = T5EncoderModel.from_pretrained('t5-base', config=self.t5_config)

        
        # self.classifier = nn.Linear(self.t5_config.d_model, num_labels)

        self.classifier = nn.Sequential(
            nn.Linear(self.t5_config.d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels)
        )
    

    def forward(self, input_ids, attention_mask):
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        cls_output = sequence_output.mean(dim=1)
        
        logits = self.classifier(cls_output)
        
        return logits
    


class TextEncoderCLIP(nn.Module):

    def __init__(self, num_labels=20): # If pretraining with PTC, change num_labels to 14
        super(TextEncoderCLIP, self).__init__()
        

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        text_model_dim = self.model.config.text_config.hidden_size
        
        self.classifier = nn.Linear(text_model_dim, num_labels)


    
    def forward(self, input_ids, attention_mask):

        text_features = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(text_features)
        
        return logits
    


class TextEncoderElectra(nn.Module):

    def __init__(self):
        super(TextEncoderElectra, self).__init__()
        
        self.electra_config = ElectraConfig.from_pretrained('google/electra-small-discriminator', 
                                                            output_hidden_states=True, 
                                                            num_hidden_layers=10)
        
        
        self.model = ElectraModel.from_pretrained('google/electra-small-discriminator', config=self.electra_config)
        
        
        # self.classifier = nn.Linear(self.electra_config.hidden_size, 14)

        self.classifier = nn.Sequential(
            nn.Linear(self.electra_config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 20) # If pretraining with PTC, change to 14
        )

    def forward(self, input_ids, attention_mask):
        
        outputs = self.model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0][:,0,:]
       
        logits = self.classifier(sequence_output)
        
        return logits
    



class TextEncoderXLNet(nn.Module):

    def __init__(self, num_labels=20): # If pretraining with PTC, change num_labels to 14
        super(TextEncoderXLNet, self).__init__()
        
        self.xlnet_config = XLNetConfig.from_pretrained('xlnet-base-cased', 
                                                        output_hidden_states=True, 
                                                        num_hidden_layers=12)
        
        
        self.model = XLNetModel.from_pretrained('xlnet-base-cased', config=self.xlnet_config)
        
       
        # self.classifier = nn.Linear(self.xlnet_config.hidden_size, num_labels)

        self.classifier = nn.Sequential(
            nn.Linear(self.xlnet_config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels) 
        )

    def forward(self, input_ids, attention_mask):
       
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        
        cls_output = sequence_output[:, -1, :]
       
        logits = self.classifier(cls_output)  
        
        return logits