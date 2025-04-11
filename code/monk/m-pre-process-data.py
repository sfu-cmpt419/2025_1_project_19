import os
import shutil
import pandas as pd

# Define paths
base_dir = "../../datasets/monk/mst-e_data/"  # Replace with your actual path
target_dir = os.path.join("../../datasets/monk/", "images")
os.makedirs(target_dir, exist_ok=True)

# CREATE MASTER FOLDER: Loop over all subject folders
for i in range(19):  # Adjust if you have more/less subjects
    subject_folder = os.path.join(base_dir, f"subject_{i}")
    if not os.path.isdir(subject_folder):
        continue
    for file in os.listdir(subject_folder):
        if file.endswith(".mp4"):
            continue  # Skip .mp4 files
        src_path = os.path.join(subject_folder, file)
        dst_path = os.path.join(target_dir, file)
        shutil.copy2(src_path, dst_path)  # or use shutil.move if you want to move instead
print(f"✅ All images collected in: {target_dir}")

# CLEAN CSV FILE
csv_path = os.path.join(base_dir, "mst-e_image_details.csv")
df = pd.read_csv(csv_path)

# Keep only 'image_ID' and 'MST' columns
df_filtered = df[["image_ID", "MST"]]
df_filtered = df_filtered[~df_filtered["image_ID"].str.endswith(".mp4")]
df_filtered = df_filtered[df_filtered["image_ID"].str.endswith(".jpg")]

# Save the filtered CSV
output_csv_path = os.path.join("../../datasets/monk/", "monk-preprocessed.csv")
df_filtered.to_csv(output_csv_path, index=False)
print(f"💾 Saved cleaned CSV to: {output_csv_path}")