import json
import numpy as np
from torch.utils.data import Dataset



def read_classes(file_path):

    classes = []
    class_defs = []


    with open(file_path, 'r', encoding='utf8') as f:
    
        for label in f.readlines():
            label = label.strip()

            if label:
                classes.append(label)

    return classes




class Task1Dataset(Dataset):

    def __init__(self, path, class_list, is_unlabeled=False, is_train=False):

        """
        path: path to the json file containing data samples 
        class_list: path to the text file containing list of possible classes for task 1
        is_unlabeled: boolean, determines if the given dataset contains the gold labels
        is_train: boolean, determines if the dataset is a train partition or not
        """


        self.path = path
        self.is_train = is_train
        self.class_list= read_classes(class_list)
        self.is_unlabeled = is_unlabeled
        

        with open(self.path, 'r') as file:
            self.data = json.load(file)



    def __getitem__(self, index):
        
        sample = self.data[index]

        sample_id = sample['id']
        text = sample['text']
        

        if not self.is_unlabeled:
            labels = sample['labels']
            labels_id = [self.class_list.index(x) for x in labels]

        else:
            labels_id = None



        if self.is_train:
            return text, labels_id


        return sample_id, text, labels_id



    def __len__(self):
        return len(self.data)
        
