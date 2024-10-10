import cv2
import os
import re
import numpy as np
from torch.utils.data import Dataset
import torch
import glob
import random



class XYDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.transpose(torch.tensor(self.x_data[idx]).float(), 0, 2)
        #y_binary = rgb_to_binary(self.y_data[idx], color_dict)
        y = torch.transpose(torch.tensor(self.y_data[idx]).float(), 0, 2)
        return x, y


def load_dataset(root, val_split_ratio=0.9):

    return x_train, y_train, x_val, y_val
