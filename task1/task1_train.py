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

import task1_dataset
from task1_dataset import Task1Dataset
from task1_collate import Collate
from task1_models import TextEncoderBERT, TextEncoderRoBERTa, TextEncoderT5, TextEncoderCLIP, TextEncoderElectra, TextEncoderXLNet
from losses import WeightedBCELoss, FocalLoss

import task1_evaluation


import sys
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

    task1_classlist = 'task1/task1_class_list.txt'

    traindata_path = 'semeval2024_dev_release/subtask1/train.json'
    train_dataset = Task1Dataset(traindata_path, task1_classlist, is_train=True, is_unlabeled=False)

    

    valdata_path = 'semeval2024_dev_release/subtask1/validation.json'
    val_dataset = Task1Dataset(valdata_path, task1_classlist, is_unlabeled=False)



    collate_fn = Collate(task1_classlist, is_unlabeled=False)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=mp.cpu_count(), collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=mp.cpu_count(), collate_fn=collate_fn)



    model_name = 'roberta'
    model = TextEncoderRoBERTa()



    '''
    If using a model pretrained with PTC, please use the below
    '''
    # model_ckpt_path = 'ptc_corpus/ptc_checkpoints/run_2024-04-29_13-26/xlnet_0.3578.pth'


    # # Load the checkpoint
    # checkpoint = torch.load(model_ckpt_path, map_location='cpu')

    # # Check if the model was saved with DataParallel or DistributedDataParallel
    # if list(checkpoint.keys())[0].startswith('module.'):
    #     # Create a new state dict without the 'module.' prefix
    #     new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    # else:
    #     new_state_dict = checkpoint


    # # Load the modified state dict into your model
    # model.load_state_dict(new_state_dict, strict=False)

    # # Modify the classifier for 20 classes
    # model.classifier = nn.Sequential(
    #         nn.Linear(model.xlnet_config.hidden_size, 512),
    #         nn.ReLU(),
    #         nn.Dropout(0.1),
    #         nn.Linear(512, 20)
    #     )



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
    run_dir = f"task1_checkpoints/run_{date_time_str}"

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)


    losses = []
    num_epochs = 20
    best_micro_f1 = 0
    best_epoch = 0
    best_checkpoint = ""

    
    for epoch in range(num_epochs):

        model.train()

        steps = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)

      
        for batch in progress_bar:

            ids, text_input, attention_masks, labels_output = batch  

            
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

        if metrics['h_f1_score = '] > best_micro_f1 : #and metrics['h_f1_score = '] >= 0.58

            best_micro_f1 = metrics['h_f1_score = ']
            best_epoch = epoch


            torch.save({
                'state_dict': model.state_dict(),
                'optimizer_dict': optimizer.state_dict(),
                'epoch': best_epoch,
            }, run_dir +'/'+ model_name + '_' + f"{best_micro_f1:.4f}" + '.pth')

            best_checkpoint = run_dir +'/'+ model_name + '_' + f"{best_micro_f1:.4f}" + '.pth'


            lines_to_write = ["=========================== Epoch: ", str(epoch+1), " | Loss: ", str(mean_loss), "===========================\n", 
                              "Validation Metrics: ", str(metrics), "\n"]

            with open(run_dir +'/best.txt', 'w') as f:
                f.writelines(lines_to_write)
                

        scheduler.step()



    
    # testing
        
    devdata_path = 'semeval2024_dev_release/dev_gold_labels/dev_subtask1_en.json'
    dev_dataset = Task1Dataset(devdata_path, task1_classlist)

    dev_dataloader = DataLoader(dev_dataset, batch_size=8, shuffle=False, num_workers=mp.cpu_count(), collate_fn=collate_fn)

    
    
    test_run_dir = f"task1_evaluation/run_{date_time_str}"

    if not os.path.exists(test_run_dir):
        os.makedirs(test_run_dir)


    
    test_metrics = validate(dev_dataloader, model, test_run_dir)

    print("Test: ", test_metrics, "\n")


    
    test_lines_to_write = ["Test: ", str(test_metrics), "\n"]

    with open(run_dir +'/test_results.txt', 'w') as f:
        f.writelines(test_lines_to_write)



    # test on unlabeled data
    # task1_evaluation.evaluate(best_checkpoint)




def validate(val_dataloader, model, run_dir):

    '''
    val_dataloader: the dataloader for the validation/testing dataset
    model: the model being evaluated
    run_dir: path to the directory used to save the results
    '''
    
    model.eval()

    predictions = []
    true_labels = []
    ids_list = []

    pred_list = []
    gold_list = []

    metrics = {}

    valdata_classlist = 'task1/task1_class_list.txt'
    classes_list = task1_dataset.read_classes(valdata_classlist)

    progress_bar = tqdm(val_dataloader, desc='Validation', leave=False)

    for batch in progress_bar:

        ids, text_input, attention_masks, labels_output = batch 

        if torch.cuda.is_available():
            text = text_input.cuda()
            labels = labels_output.cuda()
            attention_masks = attention_masks.cuda()

        with torch.no_grad():
            pred_probs = model(text, attention_masks)
            pred_classes = (pred_probs > 0.3).long()
            

        predictions.extend(pred_classes.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        ids_list.extend(ids)

   
    preds = [set(classes_list[i] for i, label in enumerate(sample) if label == 1) for sample in predictions]
    gts = [set(classes_list[i] for i, label in enumerate(sample) if label == 1) for sample in true_labels]


    for id, pred in zip(ids_list, preds):    
        pred_list.append({'id': id, 'labels': list(pred)})

    preds_json = run_dir +'/preds.json'
    with open(preds_json, 'w') as f:
        json.dump(pred_list, f)


    
    for id, gt in zip(ids_list, gts):    
        gold_list.append({'id': id, 'labels': list(gt)})
    
    gt_json = run_dir +'/gold.json'
    with open(gt_json, 'w') as f:
        json.dump(gold_list, f)


    

    macro_f1, micro_f1 = evaluate(preds, gts, classes_list)
            
    precision, recall, f1 = subtask_1_2a.evaluate_h(preds_json, gt_json)

    metrics['macroF1 = '] = macro_f1
    metrics['microF1 = '] = micro_f1

    metrics['h_precision = '] = precision
    metrics['h_recall = '] = recall
    metrics['h_f1_score = '] = f1


    return metrics






if __name__ == '__main__':
    train()