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
from src.preprocess import interpolate, crop

class FungiDataset(Dataset):
    def __init__(self, image_dir, labels_path, train, pre_load=True, train_val_split = 0.2):
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

        (self.data, self.targets) = self._load_from_disk()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.pre_load:
            return self.data[idx], self.targets[idx]
        else:
            img_path = self.data[idx]
            img = Image.open(img_path)

            # resize images to 224x224 
            # TODO replace with the desired transformation
            # I just used this so i can stack the images
            img = transforms.Resize((224, 224))(img)

            return transforms.ToTensor()(img), self.targets[idx]
            

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
                    img = img.transpose(1, 0, 2)
                    print(f"Input shape: {img.shape}")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if img is not None:
                        # Crop
                        cropped_img = crop(img, 16)

                        # Interpolation
                        scale_factor = 300 / cropped_img.shape[0]
                        sigma = scale_factor * 0.5
                        cropped_img = interpolate(cropped_img, 5, sigma)
                        cropped_img = cropped_img.transpose(1, 2, 0)

                        # Retransform 
                        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
                        cropped_img = cropped_img.transpose(1, 0, 2)

                        # Adding to array for saving as .npy
                        print(f"Output shape: {cropped_img.shape}")
                        images.append(cropped_img)
                        images = np.array(images)

                    # resize images to 224x224 
                    # TODO replace with the desired transformation
                    # I just used this so i can stack the images
                    # img = transforms.Resize((224, 224))(img)

                else:
                    images.append(img_path)

        # convert images and targets to tensors
        if self.pre_load:
            images_tensor = torch.stack([torch.from_numpy(img) for img in images])
        else:
            images_tensor = images

        targets_tensor = torch.tensor(
            metadata[['class_id', 'poisonous']].values, dtype=torch.long
        )

        # split data into training and validation sets
        train_data, val_data, train_labels, val_labels = train_test_split(images_tensor, targets_tensor, test_size=self.train_val_split, random_state=42)

        return (train_data, train_labels) if self.train else (val_data, val_labels)

    def _data_loader(self):
        train_dataset = FungiDataset(labels_path=self.labels_path, image_dir=self.image_dir, train=True, pre_load=False)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)

        val_dataset = FungiDataset(labels_path=self.labels_path, image_dir=self.image_dir, train=False, pre_load=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        return train_loader, val_loader
