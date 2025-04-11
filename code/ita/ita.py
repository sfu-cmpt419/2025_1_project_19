import os
import cv2 as cv
import numpy as np
import pandas as pd

# Define the folders and labels
folder_label_map = {
    'Black': '../../datasets/small-kaggle/Black/',
    'Brown': '../../datasets/small-kaggle/Brown/',
    'White': '../../datasets/small-kaggle/White/'
}
out_path = '../../outputs/ita-outputs/'

# -- helper functions -- #

def compute_ita(l_values, b_values):
    # Compute ITA
    epsilon = 1e-6 # avoid division by zero
    ita = np.arctan((l_values - 50) / (b_values + epsilon)) * (180 / np.pi)
    return ita

def ita_to_fitzpatrick(ita):
    if ita > 55:
        return 'I'
    elif ita > 41:
        return 'II'
    elif ita > 28:
        return 'III'
    elif ita > 10:
        return 'IV'
    elif ita > -30:
        return 'V'
    else:
        return 'VI'

def ita_to_monk_scale(ita, min_ita=-60, max_ita=90):
    ita_clipped = np.clip(ita, min_ita, max_ita)
    return int(np.round(1 + 9 * (max_ita - ita_clipped) / (max_ita - min_ita)))

# Store results here
results = []

# Process each folder
for label, path in folder_label_map.items():
    image_files = [f for f in os.listdir(path) if f.endswith('.jpg')]

    for filename in image_files:
        try:
            image_path = os.path.join(path, filename)
            img_bgr = cv.imread(image_path)

            if img_bgr is None:
                print(f"⚠️ [WARNING] Unable to read {filename} in {label}")
                continue

            img_lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)

            # Simple skin mask based on L*
            skin_mask = img_lab[:, :, 0] > 30

            l_vals = img_lab[:, :, 0][skin_mask]
            b_vals = img_lab[:, :, 2][skin_mask]

            if len(l_vals) == 0 or len(b_vals) == 0:
                print(f"⚠️ [WARNING] No skin pixels found in {filename}")
                continue

            l_mean = np.mean(l_vals)
            b_mean = np.mean(b_vals)

            ita = compute_ita(l_mean, b_mean)

            # conversions
            fitzpatrick = ita_to_fitzpatrick(ita)
            monk_scale = ita_to_monk_scale(ita)

            results.append({
                'filename': filename,
                'label': label,
                'L*': l_mean,
                'b*': b_mean,
                'ITA': ita,
                'fitzpatrick': fitzpatrick,
                'monk': monk_scale
            })

        except Exception as e:
            print(f"⚠️ [ERROR] Processing {filename}: {e}")

# Save results
csv_path = os.path.join(out_path, 'ita-fitz-monk.csv')
df = pd.DataFrame(results)
df.to_csv(csv_path, index=False)

print(f"✅ [DONE] ITA values saved to '{out_path}ita-fitz-monk.csv'")
