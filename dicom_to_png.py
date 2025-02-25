import pydicom
import numpy as np
from PIL import Image
import os
import shutil

def dicom_to_png(dicom_path, output_path):
    """Convert one single DICOM image to PNG format."""
    dicom_image = pydicom.dcmread(dicom_path)
    image_array = dicom_image.pixel_array

    # normalize to 0-255
    image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255.0
    image_array = image_array.astype(np.uint8)

    # convert to PIL and save as PNG
    image = Image.fromarray(image_array)
    image.save(output_path)

def process_dicom_folder(input_root, output_root):
    """Walk through subfolders, convert DICOM images and copy index file."""
    for subdir, _, files in os.walk(input_root):
        relative_path = os.path.relpath(subdir, input_root)
        output_subdir = os.path.join(output_root, relative_path)

        os.makedirs(output_subdir, exist_ok=True)

        for file in files:
            input_file_path = os.path.join(subdir, file)

            if file.endswith(".dcm"):
                # convert DICOM to PNG
                output_file_path = os.path.join(output_subdir, file.replace(".dicom", ".png"))
                dicom_to_png(input_file_path, output_file_path)

            elif file == "index.html":
                # copy index.html
                shutil.copy(input_file_path, output_subdir)

# input and output folders
in_dir = "../images"
out_dir = "../png_images" 

process_dicom_folder(in_dir, out_dir)

print("Done!")
