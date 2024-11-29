import os
import cv2
import numpy as np
import time
import psutil  # For measuring memory usage
import torchvision
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

input_folder = "data/x_train"
output_folder = "data/train_np"
test_folder = "data/test"

def process(img):
    images = []
    img = img.transpose(1, 0, 2)
    # print(f"Input shape: {img.shape}")
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
        # print(f"Output shape: {cropped_img.shape}")
        return np.array(cropped_img)


def preprocess(folder_path, output_folder, save_numpy=False):
    processed_images = []

    # Measure start time
    start_time = time.time()

    # Initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".PNG", ".JPG")):
            img = cv2.imread(os.path.join(folder_path, filename))
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
                processed_images.append(cropped_img)

                if not save_numpy:
                    # Save the cropped image to the output folder as individual images
                    output_path = os.path.join(output_folder, f"c_{filename}")
                    cv2.imwrite(output_path, cropped_img)
                    print(f"Saved image: {output_path}")
            else:
                print(f"Failed to read image: {filename}")
        else:
            print(f"Ignoring non-image file: {filename}")

    if save_numpy:
        # Save the entire batch of processed images as a .npy file
        processed_images = np.array(processed_images)
        np.save(os.path.join(output_folder, 'x_train.npy'), processed_images)
        print(f"Saved processed images as numpy array to: {os.path.join(output_folder, 'x_train.npy')}")

    end_time = time.time()
    
    # final memory usage
    final_memory = process.memory_info().rss / (1024 ** 2)  # to MB

    # Print metrics
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Memory used: {final_memory - initial_memory:.2f} MB")

def crop(img, crop_s = 16):
    w, h, c = img.shape

    # Crop x pixels from the top and bottom
    if h > 270:
        img = img[:, crop_s:h - crop_s, :]  
    #print(img.shape)
    return img

def interpolate(img, kernel_size = 5, sigma = 0.1, int_mode = "bilinear"):
    img = torch.tensor(img)
    # Blur for noise reduc
    blur = torchvision.transforms.GaussianBlur(kernel_size, sigma)
    blured_img = blur(img)
    blured_img = blured_img.transpose(0, 2)
    blured_img = blured_img.transpose(1, 2)
    # print(blured_img.shape)
    size = [224, 224]
    interpolated_img = F.interpolate(blured_img.unsqueeze(0), size, mode= int_mode)
    interpolated_img = interpolated_img.squeeze(0)
    # print(interpolated_img.shape)
    return interpolated_img.detach().numpy()

def downsample(data):
    d1 = cv2.pyrDown(data)
    d2 = cv2.pyrDown(d1)
    # d3 = cv2.pyrDown(d2)
    print("Original shape: ", data.shape, "Downsampled shape: ", d2.shape)
    return np.array(d2)

def upsample(data):
    d1 = cv2.pyrUp(data)
    d2 = cv2.pyrUp(d1)
    # d3 = cv2.pyrDown(d2)
    print("Original shape: ", data.shape, "Downsampled shape: ", d2.shape)
    return np.array(d2)

def test_np_load(np_path):
    images = np.load(np_path)
    print(images.shape)

    def visualize_images(images, num_images=2):
        plt.figure(figsize=(6, 3))
    
        for i in range(min(num_images, images.shape[0])):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(images[i].astype('uint8')) 
            plt.axis('off')
        plt.show()

    # visualize_images(images, num_images=2)

# ----------------------------------------------------------------------------

# preprocess(test_folder, test_folder, save_numpy=False)
# test_np_load(test_folder)

# preprocess(input_folder, output_folder, save_numpy=True)
# test_np_load('data/train_np/x_train.npy')


