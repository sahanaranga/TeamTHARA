import numpy as np
import pandas as pd
from torch.utils.data import Dataset



def read_classes(file_path):

    classes = []

    with open(file_path, 'r', encoding='utf8') as f:
    
        for label in f.readlines():
            label = label.strip()

            if label:
                classes.append(label)
        
    return classes



class PTCDataset(Dataset):

    def __init__(self, data):
        
        self.df = pd.read_csv(data)
        self.text = self.df.iloc[:, 0]
        self.label = self.df.iloc[:, 1]

        self.class_list = read_classes("ptc_corpus/classlist.txt")


    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):

        text = self.text[index]
        label = [self.label[index]]

        labels_id = [self.class_list.index(x) for x in label]

        return text, labels_id