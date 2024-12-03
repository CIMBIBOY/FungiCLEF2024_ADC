import cv2
import os
import re
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import glob
import random
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from src.preprocess import process

class FungiDataset(Dataset):
    def __init__(self, image_dir, labels_path, train, pre_load=True, augment=False ,train_val_split = 0.2, batch_size = 32):
        '''
        Args:
            image_dir: directory containing the images
            labels_path: path to the labels csv file
            train: True if training, False if validation
            pre_load: True if images should be loaded into memory, False otherwise
        '''

        self.train = train
        self.pre_load = pre_load
        self.train_val_split = train_val_split
        self.image_dir = image_dir
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.augment = augment
        self.transform = self.get_transform()
        (self.data, self.targets) = self._load_data()
        
        # Create DataLoader
        self.loader = DataLoader(self, batch_size=batch_size, shuffle=self.train, num_workers=0)

    def __len__(self):
        return len(self.data)
    
    def get_transform(self):
        if self.augment:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.ToTensor()
            ])
        else:
            return transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.ToTensor()
            ])

    def __getitem__(self, idx):
        if self.pre_load:
            target = self.targets[idx]
            image= self.data[idx]

            dict = {
                    "image": image,
                    "target_sem_cls": target[:-1],
                    "target_poisonous": target[-1]
                    }
            
            return dict
        else:
            img_path = self.data[idx]
            full_path = os.path.join(self.image_dir, img_path)
            img = cv2.imread(full_path)
            # Crop the image
            np_img = process(img)
            image = torch.from_numpy(np_img).float()
            # Normalize the data
            image = image / 255.0
            target = self.targets[idx]

           
            
                  
                    
            
            return image,(target[:-1],target[-1])
            
            
    def _load_data(self):
        # csv containing the labels
        metadata = pd.read_csv(self.labels_path)

        # image files in the image directory
        image_files = image_files = {f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg'))}
        
        # filter dataframe to include only existing images
        metadata = metadata[metadata['image_path'].isin(image_files)]
        if metadata.empty:
            raise ValueError('No matching images found in the image directory')
        
        #Order the files in the same order as the metadata
        image_files = metadata['image_path'].values
        labels= metadata[['class_id', 'poisonous']].values

        #Reindex class ids
        class_ids = labels[:,0]
        unique_class_ids = np.unique(class_ids)
        class_id_map = {class_id: idx for idx, class_id in enumerate(unique_class_ids)}
        labels[:,0] = [class_id_map[class_id] for class_id in class_ids]

        # split data into training and validation sets
        train_data_image_path, val_data_image_path, train_labels, val_labels = train_test_split(image_files, labels , test_size=self.train_val_split, random_state=42)

        train_data = train_data_image_path
        val_data = val_data_image_path

        if self.pre_load:
            if self.train:
                train_data = self._load_from_disk(train_data_image_path)
                #Normalize the data

            else:
                val_data = self._load_from_disk(val_data_image_path)

        data= train_data if self.train else val_data
        labels = train_labels if self.train else val_labels

        class_ids = labels[:,0]
        poisonous = labels[:,1]
        # Convert class ids to one-hot encoding
        class_ids = np.eye(len(np.unique(class_ids)))[class_ids]
        
        labels=torch.tensor(np.concatenate((class_ids, poisonous.reshape(-1,1)), axis=1))
        
        return data , labels
    
        

    def _load_from_disk(self,image_files):
        '''
        Load and match image files with their corresponding CSV entries. 
        Matching is necessary beacuse the image directory and the CSV file may not contain the data in the same order.

        Returns:
            data: tensor of image data if pre_load is True, otherwise tensor of image paths
            targets: tensor of class and poisonous labels
        '''
        images=[]
        for image_file in image_files:
            img_path = cv2.imread(os.path.join(self.image_dir, image_file))    
            images.append(process(img_path)).float()


        images_tensor = torch.stack([torch.from_numpy(img) for img in images])
        print(images_tensor.shape)
        # Normalize the data
        images_tensor = images_tensor / 255.0



        return images_tensor

    def get_loader(self):
        ''' Return the DataLoader for this dataset. '''
        return self.loader
    
#'''  
# Testing DataLoader

# Create dataset for training
#train_dataset = FungiDataset(image_dir="/Users/czimbermark/Documents/Egyetem/Adatelemzes/Nagyhazi/FungiCLEF2024_ADC/data/x_train", labels_path="/Users/czimbermark/Documents/Egyetem/Adatelemzes/Nagyhazi/FungiCLEF2024_ADC/data/train_metadata_height.csv", train=True, pre_load=True, batch_size=32)
#val_dataset = 0 # Ugyanez train = 0
# Retrieve DataLoader
#train_loader = train_dataset.get_loader()

# Iterate through the DataLoader
#for batch_data, batch_targets in train_loader:
#   print(f"Batch data shape: {batch_data.shape}, Batch targets: {batch_targets}")
    
#'''

def build_dataset(args):
    train_dataset = FungiDataset(image_dir=args["image_dir"], labels_path=args["labels_path"], train=True, pre_load=args["pre_load"], batch_size=args["batch_size"])
    val_dataset =  FungiDataset(image_dir=args["image_dir"], labels_path=args["labels_path"], train=False, pre_load=args["pre_load"], batch_size=args["batch_size"])
    # Retrieve DataLoader
    train_loader = train_dataset.get_loader()
    valid_loader = val_dataset.get_loader()
    return train_loader, valid_loader