import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
import torch

path_4_3 = "data/4_3_images"
path_4_3_2 = "data/4032x3024_images"
path_3_2 = "data/3_2_images"

path_korlap3 = "data/korlap3"

output_folder = "data/final_trunks"
# output2_folder = "/Users/czimbermark/Documents/Deep_Learning/agriculture-image-processing/tree_trunk_segmentation/data/cropped_korlap3"

test = "/Users/czimbermark/Documents/Deep_Learning/agriculture-image-processing/tree_trunk_segmentation/data/test"

def crop_and_save_images(folder_path, output_folder, target_ratio=(4000, 3000)):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".PNG", ".JPG")):
            img = cv2.imread(os.path.join(folder_path, filename))
            img = img.transpose(1, 0, 2)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is not None:
                cropped_img = crop(img, target_ratio)
                scale_factor = 4000/cropped_img.shape[0] 
                sigma = scale_factor * 0.5 
                cropped_img = interpolate(cropped_img, 5, sigma)
                cropped_img = cropped_img.transpose(1, 2, 0)
                print(cropped_img.shape)
                # downsampled_img = downsample(cropped_img)
                # print("downsampled")
                
                #retransform 
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
                cropped_img = cropped_img.transpose(1, 0, 2)
                print("retransformed")
                # Save the cropped image to the output folder
                output_path = os.path.join(output_folder, f"c_{filename}")
                print(cropped_img.shape)
                cv2.imwrite(output_path, cropped_img)
            else:
                print(f"Failed to read image: {filename}")
        else:
            print(f"Ignoring non-image file: {filename}")

def crop(img, target_ratio=(4000, 3000)):
    w, h, c = img.shape
    current_ratio = w / h  # Calculate the current aspect ratio

    if current_ratio > target_ratio[0] / target_ratio[1]:
        # Crop the width to achieve the target aspect ratio
        crop_width = int(h * target_ratio[0] / target_ratio[1])
        crop_start = (w - crop_width) // 2
        cropped_img = img[crop_start:crop_start + crop_width, :, :]

    else:
        # Crop the height to achieve the target aspect ratio
        crop_height = int(w * target_ratio[1] / target_ratio[0])
        crop_start = (h - crop_height) // 2
        cropped_img = img[:, crop_start:crop_start + crop_height, :]

    print(cropped_img.shape)

    return np.array(cropped_img)

def interpolate(img, kernel_size = 5, sigma = 0.1):
    img = torch.tensor(img)
    blur = torchvision.transforms.GaussianBlur(kernel_size, sigma)
    blured_img = blur(img)
    blured_img = blured_img.transpose(0, 2)
    blured_img = blured_img.transpose(1, 2)
    print(blured_img.shape)
    size = [4000, 3000]
    interpolated_img = F.interpolate(blured_img.unsqueeze(0), size, mode= "bilinear")
    interpolated_img = interpolated_img.squeeze(0)
    print(interpolated_img.shape)
    return interpolated_img.detach().numpy()

def downsample(data):
    d1 = cv2.pyrDown(data)
    d2 = cv2.pyrDown(d1)
    # d3 = cv2.pyrDown(d2)
    print("Original shape: ", data.shape, "Downsampled shape: ", d2.shape)
    return np.array(d2)

crop_and_save_images(path_4_3, output_folder)
crop_and_save_images(path_4_3_2, output_folder)
crop_and_save_images(path_3_2, output_folder)
crop_and_save_images(path_korlap3, output_folder)


from PIL import Image
import os

def crop_and_save_images(input_dir, output_dir, crop_height_start=1125, crop_height_end=1875):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # List all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".JPG") or filename.endswith(".png"):  # Add more conditions if needed
            file_path = os.path.join(input_dir, filename)
            with Image.open(file_path) as img:
                # Crop the image to the desired sub-region
                crop_box = (0, crop_height_start, 4000, crop_height_end)
                cropped_img = img.crop(crop_box)
                
                # Save the cropped image to the output directory
                output_file_path = os.path.join(output_dir, filename)
                cropped_img.save(output_file_path)
                print(f"Processed and saved: {output_file_path}")

# Usage
input_directory = 'data/forest_depth'
output_directory = 'data/final_trunk_middles_l'
crop_and_save_images(input_directory, output_directory)

input_directory = 'data/final_trunks'
output_directory = 'data/final_trunk_middles'
crop_and_save_images(input_directory, output_directory)