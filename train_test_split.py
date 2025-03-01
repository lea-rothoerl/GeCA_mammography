import os
import pandas as pd
import shutil

# Paths
lesion_dir = "../lesions_png"
csv_path = "../../shared_data/VinDr_Mammo/finding_annotations.csv"

# Load annotations CSV
df = pd.read_csv(csv_path)

# Create output directories
train_dir = os.path.join(lesion_dir, "training")
test_dir = os.path.join(lesion_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Dictionary mapping image_id to split type
split_dict = dict(zip(df["image_id"], df["split"]))

# Process lesion files
for filename in os.listdir(lesion_dir):
    if filename.endswith(".png"):
        image_id = filename.split("_lesion_")[0]

        if image_id in split_dict:
            split = split_dict[image_id]
            if split == "training":
                dest_dir = train_dir
            elif split == "test":
                dest_dir = test_dir
            else:
                print(f"Split unknown for {filename}, skipping...")
                continue

            src_path = os.path.join(lesion_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            
            shutil.move(src_path, dest_path) 
            print(f"Moved {filename} to {dest_dir}/")

print("Finished moving lesion images.")
