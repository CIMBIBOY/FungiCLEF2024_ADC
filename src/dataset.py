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
from preprocess import process

class FungiDataset(Dataset):
    def __init__(self, image_dir, labels_path, train, pre_load=True, train_val_split = 0.2, batch_size = 32):
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

        (self.data, self.targets) = self._load_from_disk()
        # Create DataLoader
        self.loader = DataLoader(self, batch_size=batch_size, shuffle=self.train, num_workers=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.pre_load:
            return self.data[idx], self.targets[idx]
        else:
            img_path = self.data[idx]
            img = cv2.imread(img_path)
            np_img = process(img)

            return torch.from_numpy(np_img), self.targets[idx]
            

    def _load_from_disk(self):
        '''
        Load and match image files with their corresponding CSV entries. 
        Matching is necessary beacuse the image directory and the CSV file may not contain the data in the same order.

        Returns:
            data: tensor of image data if pre_load is True, otherwise tensor of image paths
            targets: tensor of class and poisonous labels
        '''
        # csv containing the labels
        metadata = pd.read_csv(self.labels_path)

        # image files in the image directory
        image_files = image_files = {f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg'))}
        
        # filter dataframe to include only existing images
        metadata = metadata[metadata['image_path'].isin(image_files)]

        if metadata.empty:
            raise ValueError('No matching images found in the image directory')

        # load images if pre_load is True, otherwise store image paths
        images = []
        for image_path in metadata['image_path']:
            if image_path.lower().endswith((".jpg", ".png", ".PNG", ".JPG")):
                img_path = cv2.imread(os.path.join(self.image_dir, image_path))

                if self.pre_load:
                    images.append(process(img_path))

                else:
                    images.append(img_path)
    
        # convert images and targets to tensors
        if self.pre_load:
            images_tensor = torch.stack([torch.from_numpy(img) for img in images])
            print(images_tensor.shape)
        else: 
            images_tensor = images

        targets_tensor = torch.tensor(
            metadata[['class_id', 'poisonous']].values, dtype=torch.long
        )

        # split data into training and validation sets
        train_data, val_data, train_labels, val_labels = train_test_split(images_tensor, targets_tensor, test_size=self.train_val_split, random_state=42)

        return (train_data, train_labels) if self.train else (val_data, val_labels)

    def get_loader(self):
        ''' Return the DataLoader for this dataset. '''
        return self.loader
    
#'''  
# Testing DataLoader

# Create dataset for training
train_dataset = FungiDataset(image_dir="/Users/czimbermark/Documents/Egyetem/Adatelemzes/Nagyhazi/FungiCLEF2024_ADC/data/x_train", labels_path="/Users/czimbermark/Documents/Egyetem/Adatelemzes/Nagyhazi/FungiCLEF2024_ADC/data/train_metadata_height.csv", train=True, pre_load=True, batch_size=32)
val_dataset = 0 # Ugyanez train = 0
# Retrieve DataLoader
train_loader = train_dataset.get_loader()

# Iterate through the DataLoader
for batch_data, batch_targets in train_loader:
    print(f"Batch data shape: {batch_data.shape}, Batch targets: {batch_targets}")
    
#'''