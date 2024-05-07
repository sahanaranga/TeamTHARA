import os
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

import dataset
from dataset import PTCDataset
from collate import Collate


import sys
sys.path.append('task1')
import task1_models
from task1_models import TextEncoderBERT, TextEncoderRoBERTa, TextEncoderT5, TextEncoderCLIP, TextEncoderElectra, TextEncoderXLNet


sys.path.append('scorer-baseline')
import subtask_1_2a






def evaluate(pred, gold, CLASSES):

    '''
    pred: list of predictions
    gold: list of gold labels
    CLASSES: list of possible classes 
    '''
  
    mlb = MultiLabelBinarizer()
    mlb.fit([CLASSES])

    gold = mlb.transform(gold)
    pred = mlb.transform(pred)

    macro_f1 = f1_score(gold, pred, average="macro", zero_division=1)
    micro_f1 = f1_score(gold, pred, average="micro", zero_division=1)

    return macro_f1, micro_f1



def train():

    classlist = 'ptc_corpus/classlist.txt'

    traindata_path = 'ptc_corpus/ptc_train_dataset.csv'
    train_dataset = PTCDataset(traindata_path)


    valdata_path = 'ptc_corpus/ptc_validation_dataset.csv'
    val_dataset = PTCDataset(valdata_path)


    collate_fn = Collate()

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=mp.cpu_count(), collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=mp.cpu_count(), collate_fn=collate_fn)


    model_name = 'clip'
    model = TextEncoderCLIP()
    model = torch.nn.DataParallel(model)
    model = model.cuda()



    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.AdamW([
                              {"params": model.module.model.parameters(), "lr": 0.00002},
                              {"params": model.module.classifier.parameters(), "lr": 0.0002},
                              ]
                             , lr=0.002)

    scheduler = MultiStepLR(optimizer, milestones=[4, 8], gamma=0.8) 



    
    cur_date_time = datetime.now()
    date_time_str = cur_date_time.strftime('%Y-%m-%d_%H-%M')
    run_dir = f"ptc_corpus/ptc_checkpoints/run_{date_time_str}"

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)


    losses = []
    num_epochs = 10
    best_f1 = 0
    best_epoch = 0

    
    for epoch in range(num_epochs):

        model.train()

        steps = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)

      
        for batch in progress_bar:

            text_input, attention_masks, labels_output = batch  

            
            text = text_input.cuda()
            labels = labels_output.cuda()
            attention_masks = attention_masks.cuda()

            optimizer.zero_grad()

            logits = model(text, attention_masks)
            loss = criterion(logits, labels.float()) 

            losses.append(loss.item())

            # print ("Epoch: ", epoch," | Loss: ", loss)

            loss.backward()
            optimizer.step()
            

            steps += 1



        mean_loss = torch.mean(torch.tensor(losses)).item()

        print ("=========================== Epoch: ", epoch+1," | Loss: ", mean_loss, "===========================\n")


        metrics = validate(val_dataloader, model, run_dir)
        print("Validation: ", metrics, "\n")

        if metrics['microF1 = '] > best_f1 :

            best_f1 = metrics['microF1 = ']
            best_epoch = epoch


            torch.save({
                'state_dict': model.state_dict(),
                'optimizer_dict': optimizer.state_dict(),
                'epoch': best_epoch,
            }, run_dir +'/'+ model_name + '_' + f"{best_f1:.4f}" + '.pth')


            lines_to_write = ["=========================== Epoch: ", str(epoch+1), " | Loss: ", str(mean_loss), "===========================\n", 
                              "Validation Metrics: ", str(metrics), "\n"]

            with open(run_dir +'/best.txt', 'w') as f:
                f.writelines(lines_to_write)
                

        scheduler.step()



    
    




def validate(val_dataloader, model, run_dir, is_test=False):
    
    model.eval()

    predictions = []
    true_labels = []
    ids_list = []

    pred_list = []
    gold_list = []

    metrics = {}

    valdata_classlist = 'ptc_corpus/classlist.txt'
    classes_list = dataset.read_classes(valdata_classlist)

    progress_bar = tqdm(val_dataloader, desc='Validation', leave=False)

    for batch in progress_bar:

        text_input, attention_masks, labels_output = batch 

        if torch.cuda.is_available():
            text = text_input.cuda()
            labels = labels_output.cuda()
            attention_masks = attention_masks.cuda()

        with torch.no_grad():
            pred_probs = model(text, attention_masks)
            pred_classes = (pred_probs > 0.3).long()
            

        predictions.extend(pred_classes.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    
    preds = [set(classes_list[i] for i, label in enumerate(sample) if label == 1) for sample in predictions]
    gts = [set(classes_list[i] for i, label in enumerate(sample) if label == 1) for sample in true_labels]


    macro_f1, micro_f1 = evaluate(preds, gts, classes_list)
            

    metrics['macroF1 = '] = macro_f1
    metrics['microF1 = '] = micro_f1


    return metrics






if __name__ == '__main__':
    train()