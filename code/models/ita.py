import os
import cv2 as cv
import numpy as np

# path to images
base_path = '/Users/aryaman_bahuguna/Documents/Spring2025/CMPT419/project/2025_1_project_19/datasets/small-kaggle/Black/'

## READ: the image (in BGR format by default) & convert to LAB
sample_image = os.path.join(base_path, '1_1_1_20170109190848182.jpg.chip.jpg')
img_bgr = cv.imread(sample_image)
img_lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB) # convert to LAB

## FILTER: out background or non-skin pixels here
# compute mean L* and b* over skin pixels only
skin_mask = img_lab[:, :, 0] > 30  # Adjust threshold as needed
l_mean = np.mean(img_lab[:, :, 0][skin_mask])
b_mean = np.mean(img_lab[:, :, 2][skin_mask])

# Reshape image to get pixels as rows
pixels_lab = img_lab.reshape(-1, 3)

# Compute mean L* and b* across the skin region
l_mean = np.mean(img_lab[:, :, 0][skin_mask])
b_mean = np.mean(img_lab[:, :, 2][skin_mask])

# COMPUTE: ITA
epsilon = 1e-6  # Small value to avoid division by zero
ita = np.arctan((l_mean - 50) / (b_mean + epsilon)) * (180 / np.pi)
print("ITA =", ita)