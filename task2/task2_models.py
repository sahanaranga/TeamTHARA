import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import BertConfig, BertModel, RobertaConfig, RobertaModel, CLIPModel, LayoutLMv2Model

import sys
sys.path.append('task1')
from task1_models import TextEncoderBERT, TextEncoderRoBERTa



class ImageEncoderConvNeXT(nn.Module):

    def __init__(self):
        super(ImageEncoderConvNeXT, self).__init__()

        
        model = models.convnext_base(pretrained=True)
        modules = list(model.children())[:-2]

        self.model = nn.Sequential(*modules)

        
        self.pool = nn.AdaptiveAvgPool2d((10, 1))
        self.linear = nn.Linear(1024, 512)  


    def forward(self, x):

        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)

        out = out.transpose(1, 2).contiguous()
        out = self.linear(out)

        return out 
    


class ImageEncoderResNet(nn.Module):

    def __init__(self):
        super(ImageEncoderResNet, self).__init__()

        model = models.resnet50(pretrained=True)
        modules = list(model.children())[:-1]  

        self.model = nn.Sequential(*modules)

        self.linear = nn.Linear(2048, 512)


    def forward(self, x):
        
        # print("x", x.shape)
        out = self.model(x)

        # print("resnet", out.shape)
        out = torch.flatten(out, start_dim=2) 

        # print("resnet 1", out.shape)
        out = out.transpose(1, 2).contiguous()
        out = self.linear(out)


        return out 
    


class ImageEncoderCLIP(nn.Module):

    def __init__(self):
        super(ImageEncoderCLIP, self).__init__()

        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.linear = nn.Linear(1024, 512)

       
    def forward(self, images):

        outputs, tokens = self.model.get_image_features(images["pixel_values"])
        outputs = self.linear(tokens)

        return outputs


    
    

class TextEncoderBERT(nn.Module):
    
    def __init__(self):
        super(TextEncoderBERT, self).__init__()

        self.bert_model = BertModel.from_pretrained('bert-base-cased')
        self.linear = nn.Linear(self.bert_model.config.hidden_size, 512)


    def forward(self, input_ids, attention_mask):

        out = self.bert_model(input_ids=input_ids, attention_mask=attention_mask) 
        out = out.last_hidden_state[:,0,:]
        out = self.linear(out)


        return out
    


class TextEncoderCLIP(nn.Module):

    def __init__(self):
        super(TextEncoderCLIP, self).__init__()
        
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        text_model_dim = self.model.config.text_config.hidden_size
        
        self.linear = nn.Linear(text_model_dim, 512)
        
    
    def forward(self, input_ids, attention_mask=None):
        
        out = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        
        out = self.linear(out)
        
        return out
    


class TextEncoderRoBERTa(nn.Module):

    def __init__(self):
        super(TextEncoderRoBERTa, self).__init__()

        
        self.roberta_config = RobertaConfig.from_pretrained('roberta-base', 
                                                            output_hidden_states=True, 
                                                            num_hidden_layers=10)
        
        
        self.model = RobertaModel.from_pretrained('roberta-base', config=self.roberta_config)

        self.linear = nn.Linear(self.roberta_config.hidden_size, 512)
    

    def forward(self, input_ids, attention_mask):
        
        # print("input roberta", input_ids.shape)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        
        out = outputs[0]
        # print("roberta out", outputs[0].shape)
        # exit(0)
        
        out = self.linear(out)
        # print("final out", out.shape)
        # exit(0)
        
        return out
    



class TransformerClassifier(nn.Module):

    def __init__(self, input_dim, num_classes):
        super(TransformerClassifier, self).__init__()

        self.linear1 = nn.Linear(input_dim, 512)
        self.transformer_block = nn.TransformerEncoderLayer(d_model=512, nhead=4)
        self.linear2 = nn.Linear(512, num_classes)


        

    def forward(self, x):

        x = self.linear1(x)
        # print(x.shape)
        x = self.transformer_block(x)
        x = self.linear2(x)

        return x
    


class FusionModel(nn.Module):

    def __init__(self, num_classes, text_ckpt=None):

        super(FusionModel, self).__init__()


        '''
        If you are not using a text encoder pretrained on PTC, please comment out lines 199-213 and use line 216 instead
        '''
        self.text_encoder = RoBERTaPTC()
        # self.text_encoder = BERTPTC()

        
        checkpoint = torch.load(text_ckpt)

       
        if list(checkpoint.keys())[0].startswith('module.'):
            new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}

        else:
            new_state_dict = checkpoint


        self.text_encoder.load_state_dict(new_state_dict, strict=False)


        # self.text_encoder = TextEncoderRoBERTa()
        self.image_encoder = ImageEncoderCLIP()

        self.transformer_classifier = TransformerClassifier(input_dim=512, num_classes=num_classes)

        
        # self.transformer_classifier = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(256, 22)
        # )


    def forward(self, input_ids, attention_mask, images):

        # print("input ids", input_ids.shape)

        text_features = self.text_encoder(input_ids, attention_mask)
        image_features = self.image_encoder(images)

        # print("text features", text_features.shape)
       
        combined_features = torch.cat((text_features, image_features), dim=1)
        # print("concat", combined_features.shape)
        # exit(0)
        
        logits = self.transformer_classifier(combined_features) 
        # logits = self.transformer_classifier(image_features) 
        # print("logits", logits.shape)
        # exit(0)

        return logits[:, 0, :]
    






class RoBERTaPTC(TextEncoderRoBERTa):

    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(self.roberta_config.hidden_size, 512)
    
    def forward(self, input_ids, attention_mask):
        
        outputs = self.model(input_ids, attention_mask=attention_mask)
        out = outputs[0]
        
        out = self.linear(out)
         
        return out
    


class BERTPTC(TextEncoderBERT):

    def __init__(self):
        super().__init__()

        self.bert_config = BertConfig.from_pretrained('bert-base-cased', 
                                                 output_hidden_states=True, 
                                                 num_hidden_layers=10)
        
        self.model = BertModel.from_pretrained('bert-base-cased', config=self.bert_config)

        self.linear = nn.Linear(self.bert_config.hidden_size, 512)
    

    def forward(self, input_ids, attention_mask):
        
        outputs = self.model(input_ids, attention_mask=attention_mask)
        out = outputs[0]
        
        out = self.linear(out)
        
        return out