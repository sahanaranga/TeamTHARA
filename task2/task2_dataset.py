import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms



def read_classes(file_path):

    classes = []

    with open(file_path, 'r', encoding='utf8') as f:
    
        for label in f.readlines():
            label = label.strip()

            if label:
                classes.append(label)
        
    return classes



class Task2Dataset(Dataset):


    def __init__(self, path, class_list, image_dir, is_train=False):

        """
        path: path to the json file containing data samples 
        class_list: path to the text file containing list of possible classes for task 1
        image_dir: path to the directory containing the images
        is_train: boolean, determines if the dataset is a train partition or not
        """

        self.path = path
        self.class_list = read_classes(class_list)
        self.image_dir = image_dir
        self.is_train = is_train

        self.preprocess = transforms.Compose([
            transforms.Resize(256),  
            transforms.CenterCrop(224),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
        ])
       

        with open(self.path, 'r') as file:
            self.data = json.load(file)



    def __getitem__(self, index):
        
        sample = self.data[index]

        sample_id = sample['id']
        text = sample['text']

        image_path = self.image_dir + '/' + sample['image']
        image = Image.open(image_path).convert("RGB")


        '''
        If using a CNN-based model, please uncomment the following line
        '''
        # image = self.preprocess(image)
        

        labels = sample['labels']
        labels_id = [self.class_list.index(x) for x in labels]


        if self.is_train:
            return  text, image, labels_id

        return sample_id, text, image, labels_id




    def __len__(self):
        return len(self.data)
        
