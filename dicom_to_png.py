import pydicom
import numpy as np
from PIL import Image
import os
import shutil

def dicom_to_png(dicom_path, output_path):
    """Convert one DICOM image to PNG."""
    try:
        dicom_image = pydicom.dcmread(dicom_path)

        # ensure the file contains pixel data
        if not hasattr(dicom_image, "pixel_array"):
            print(f"Skipping {dicom_path}: No pixel data found.")
            return

        image_array = dicom_image.pixel_array

        # normalize pixel values to 0-255
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255.0
        image_array = image_array.astype(np.uint8)

        # convert to PNG
        image = Image.fromarray(image_array)
        image.save(output_path)

    except Exception as e:
        print(f"Error processing {dicom_path}: {e}")

def process_dicom_folder(input_root, output_root):
    """Process subfolders, convert DICOM images and copy index.html."""
    for subdir, _, files in os.walk(input_root):
        relative_path = os.path.relpath(subdir, input_root)
        output_subdir = os.path.join(output_root, relative_path)

        os.makedirs(output_subdir, exist_ok=True)

        for file in files:
            input_file_path = os.path.join(subdir, file)

            # case-insensitive check for .dicom
            if file.lower().endswith(".dicom"):
                output_file_path = os.path.join(output_subdir, os.path.splitext(file)[0] + ".png")
                dicom_to_png(input_file_path, output_file_path)

            elif file == "index.html":
                shutil.copy(input_file_path, output_subdir)

in_dir = "../images"
out_dir = "../png_images"

process_dicom_folder(in_dir, out_dir)

print("Done!")
