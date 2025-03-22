import os
import cv2 as cv
import numpy as np
import pandas as pd

# Define the folders and labels
folder_label_map = {
    'Black': '/Users/aryaman_bahuguna/Documents/Spring2025/CMPT419/project/2025_1_project_19/datasets/small-kaggle/Black/',
    'Brown': '/Users/aryaman_bahuguna/Documents/Spring2025/CMPT419/project/2025_1_project_19/datasets/small-kaggle/Brown/',
    'White': '/Users/aryaman_bahuguna/Documents/Spring2025/CMPT419/project/2025_1_project_19/datasets/small-kaggle/White/'
}
out_path = '/Users/aryaman_bahuguna/Documents/Spring2025/CMPT419/project/2025_1_project_19/outputs/'

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
                print(f"Warning: Unable to read {filename} in {label}")
                continue

            img_lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)

            # Simple skin mask based on L*
            skin_mask = img_lab[:, :, 0] > 30

            l_vals = img_lab[:, :, 0][skin_mask]
            b_vals = img_lab[:, :, 2][skin_mask]

            if len(l_vals) == 0 or len(b_vals) == 0:
                print(f"Warning: No skin pixels found in {filename}")
                continue

            l_mean = np.mean(l_vals)
            b_mean = np.mean(b_vals)

            # Compute ITA
            epsilon = 1e-6
            ita = np.arctan((l_mean - 50) / (b_mean + epsilon)) * (180 / np.pi)

            results.append({
                'filename': filename,
                'label': label,
                'L*': l_mean,
                'b*': b_mean,
                'ITA': ita
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Save results
csv_path = os.path.join(out_path, 'ita-results-all.csv')
df = pd.DataFrame(results)
df.to_csv(csv_path, index=False)

print("Successfull! ITA values saved to 'ita_results_all.csv'")
