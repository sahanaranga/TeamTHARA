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

import task2_dataset
from task2_dataset import Task2Dataset
from task2_collate import Collate
from task2_models import FusionModel


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

    task2_classlist = 'task2/task2_class_list.txt'


    traindata_path = 'semeval2024_dev_release/subtask2a/train.json'
    train_img_path = 'image_data/train_images'

    train_dataset = Task2Dataset(traindata_path, task2_classlist, train_img_path, is_train=True)



    valdata_path = 'semeval2024_dev_release/subtask2a/validation.json'
    val_img_path = 'image_data/validation_images'

    val_dataset = Task2Dataset(valdata_path, task2_classlist, val_img_path, is_train=False)



    collate_fn = Collate(task2_classlist)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=mp.cpu_count(), collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=mp.cpu_count(), collate_fn=collate_fn)



    model_name = 'roberta_vit_ptc'
    model_ckpt_path = 'ptc_corpus/ptc_checkpoints/run_2024-04-27_21-19/roberta_0.3663.pth'

    model = FusionModel(num_classes=22, text_ckpt=model_ckpt_path)

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.AdamW([
                              {"params": model.module.text_encoder.parameters(), "lr": 0.00002},
                              {"params": model.module.image_encoder.parameters(), "lr": 0.00002},
                              {"params": model.module.transformer_classifier.parameters(), "lr": 0.0002},
                              ]
                             , lr=0.002)

    scheduler = MultiStepLR(optimizer, milestones=[4, 8], gamma=0.8)



    
    cur_date_time = datetime.now()
    date_time_str = cur_date_time.strftime('%Y-%m-%d_%H-%M')
    run_dir = f"task2_checkpoints/run_{date_time_str}"

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)


    losses = []
    num_epochs = 20
    best_micro_f1 = 0
    best_epoch = 0
    

    for epoch in range(num_epochs):

        model.train()

        steps = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)


        for batch in progress_bar:

            ids, text_input, attention_masks, images_input, labels_output = batch
            
            text = text_input.cuda()
            labels = labels_output.cuda()

            '''
            If using a CNN-based model, please use line 132 and comment line 133
            '''
            # images = images_input.cuda()
            images = {k: v.cuda() for k, v in images_input.items()}

            text_attention_masks = attention_masks.cuda()

            optimizer.zero_grad()

            logits = model(text, text_attention_masks, images)
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

        if metrics['h_f1_score = '] > best_micro_f1 : #and metrics['h_f1_score = '] > 0.57

            best_micro_f1 = metrics['h_f1_score = ']
            best_epoch = epoch


            torch.save({
                'state_dict': model.state_dict(),
                'optimizer_dict': optimizer.state_dict(),
                'epoch': best_epoch,
            }, run_dir +'/'+ model_name + '_' + f"{best_micro_f1:.4f}" + '.pth')


            lines_to_write = ["=========================== Epoch: ", str(epoch+1), " | Loss: ", str(mean_loss), "===========================\n", 
                              "Validation: ", str(metrics), "\n"]

            with open(run_dir +'/best.txt', 'w') as f:
                f.writelines(lines_to_write)
                

        scheduler.step()



    # testing
        
    devdata_path = 'semeval2024_dev_release/dev_gold_labels/dev_subtask2a_en.json'
    dev_img_path = 'image_data/dev_images'

    dev_dataset = Task2Dataset(devdata_path, task2_classlist, dev_img_path, is_train=False)

    dev_dataloader = DataLoader(dev_dataset, batch_size=8, shuffle=False, num_workers=mp.cpu_count(), collate_fn=collate_fn)

    
    
    test_run_dir = f"task2_evaluation/run_{date_time_str}"

    if not os.path.exists(test_run_dir):
        os.makedirs(test_run_dir)


    
    test_metrics = validate(dev_dataloader, model, test_run_dir)

    print("Test: ", test_metrics, "\n")


    
    test_lines_to_write = ["Test: ", str(test_metrics), "\n"]

    with open(run_dir +'/test_results.txt', 'w') as f:
        f.writelines(test_lines_to_write)






def validate(val_dataloader, model, run_dir):
    
    model.eval()

    predictions = []
    true_labels = []
    ids_list = []

    pred_list = []
    gold_list = []

    metrics = {}

    valdata_classlist = 'task2/task2_class_list.txt'
    classes_list = task2_dataset.read_classes(valdata_classlist)

    progress_bar = tqdm(val_dataloader, desc='Validation', leave=False)

    for batch in progress_bar:

        ids, text_input, attention_masks, images_input, labels_output = batch

        if torch.cuda.is_available():
            text = text_input.cuda()
            labels = labels_output.cuda()

            '''
            If using a CNN-based model, please use line 250 and comment line 251 
            '''
            # images = images_input.cuda()
            images = {k: v.cuda() for k, v in images_input.items()}

            text_attention_masks = attention_masks.cuda()

        with torch.no_grad():
            pred_probs = model(text, text_attention_masks, images)
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