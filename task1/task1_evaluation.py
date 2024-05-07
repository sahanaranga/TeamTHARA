import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

import task1_dataset
from task1_dataset import Task1Dataset
from task1_collate import Collate
from task1_models import TextEncoderRoBERTa

import sys
sys.path.append('scorer-baseline')
import subtask_1_2a







def split_named_arg(arg: str) -> Tuple[Optional[str], str]:

    vals = arg.split("=", 1)
    name: Optional[str]

    if len(vals) > 1:
        name = vals[0].strip()
        value = os.path.expanduser(vals[1])

    else:
        name = None
        value = vals[0]

    return name, value






def calculate_f1(pred, gold, CLASSES):
  
    mlb = MultiLabelBinarizer()
    mlb.fit([CLASSES])

    gold = mlb.transform(gold)
    pred = mlb.transform(pred)

    macro_f1 = f1_score(gold, pred, average="macro", zero_division=1)
    micro_f1 = f1_score(gold, pred, average="micro", zero_division=1)

    return macro_f1, micro_f1





def evaluate(model_path):

    """
    model_path: path to the model checkpoint
    """
    


    task1_classlist = 'task1/task1_class_list.txt'
    classes_list = task1_dataset.read_classes(task1_classlist)

    
    devdata_path = 'test_data/english/en_subtask1_test_unlabeled.json' 
    dev_dataset = Task1Dataset(devdata_path, task1_classlist, is_unlabeled=True)


    collate_fn = Collate(task1_classlist, is_unlabeled=True)

    dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=False, num_workers=mp.cpu_count(), collate_fn=collate_fn)

    


    # used to create directory for saving checkpoints
    cur_date_time = datetime.now()
    date_time_str = cur_date_time.strftime('%Y-%m-%d_%H-%M')
    run_dir = f"/task1_evaluation/run_{date_time_str}"

    # print(run_dir)

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)




    predictions = []
    true_labels = []
    ids_list = []

    pred_list = []
    gold_list = []

    metrics = {}


    progress_bar = tqdm(dev_dataloader, desc='Testing', leave=False)

    



    model_name = 'roberta'
    model = TextEncoderRoBERTa()


    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

   
    if list(checkpoint.keys())[0].startswith('module.'):
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    else:
        new_state_dict = checkpoint



   
    model.load_state_dict(new_state_dict, strict=False)

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    model.eval()


    for batch in progress_bar:

        ids, text_input, attention_masks, labels_output = batch

        if torch.cuda.is_available():
            text = text_input.cuda()
            labels = labels_output.cuda()
            text_attention_masks = attention_masks.cuda()

        with torch.no_grad():
            pred_probs = model(text, text_attention_masks)
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


    

    macro_f1, micro_f1 = calculate_f1(preds, gts, classes_list)
            
    precision, recall, f1 = subtask_1_2a.evaluate_h(preds_json, gt_json)

    metrics['macroF1 = '] = macro_f1
    metrics['microF1 = '] = micro_f1

    metrics['h_precision = '] = precision
    metrics['h_recall = '] = recall
    metrics['h_f1_score = '] = f1


    print("Test: ", metrics, "\n")


    lines_to_write = ["Test: ", str(metrics), "\n"]

    with open(run_dir +'/best.txt', 'w') as f:
        f.writelines(lines_to_write)






if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Evaluate a trained model"
    )

    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        required=True,
        type=str,
        nargs="+",
        metavar="[NAME=]PATH",
        help=(
            "Path to the model checkpoint(s). A name can be given that is displayed "
            "instead of showing the file name of the path that was given."
        ),
    )


    options = parser.parse_args()
    
    for model_path in options.model:
        name, model_path = split_named_arg(model_path)

        if name is None:
            name = model_path


    evaluate(model_path)