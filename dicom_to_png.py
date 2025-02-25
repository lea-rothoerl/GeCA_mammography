import pydicom
import numpy as np
from PIL import Image
import os
import shutil

def crop_borders(image_array, threshold=10, white_threshold=240):
    """
    Cropping image to the smallest rectangle containing all pixels with 
    intensity above threshold.

    Parameters:
    - image_array: The normalized image array.
    - threshold: Pixel intensity threshold for non-black regions.
    - white_threshold: Pixel intensity threshold for bright regions (like text).
    
    Returns:
    - Cropped image array.
    """
    # color whites black for cropping
    rem_white = image_array.copy()
    rem_white[rem_white > white_threshold] = 0

    # binary mask with values above/below threshold 
    mask = rem_white > threshold
    
    # return original image if mask empty
    if not mask.any():
        return image_array

    # get coordinates of non-black pixels
    coords = np.argwhere(mask)
    
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    
    # crop image to these coordinates
    return image_array[y0:y1+1, x0:x1+1]

def dicom_to_png(dicom_path, output_path):
    """Convert DICOM image to PNG and crop off black borders."""
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

        # remove black borders
        image_array = crop_borders(image_array)

        # convert to PIL, save as PNG
        image = Image.fromarray(image_array)
        image.save(output_path)

    except Exception as e:
        print(f"Error processing {dicom_path}: {e}")

def process_dicom_folder(input_root, output_root):
    """Process subfolders, convert DICOM images and copy index file."""
    for subdir, _, files in os.walk(input_root):
        relative_path = os.path.relpath(subdir, input_root)
        output_subdir = os.path.join(output_root, relative_path)

        os.makedirs(output_subdir, exist_ok=True)

        for file in files:
            input_file_path = os.path.join(subdir, file)

            # check for file ending
            if file.lower().endswith(".dicom"):
                output_file_path = os.path.join(output_subdir, os.path.splitext(file)[0] + ".png")
                dicom_to_png(input_file_path, output_file_path)

            elif file == "index.html":
                shutil.copy(input_file_path, output_subdir)

in_dir = "../images"
out_dir = "../png_images"

process_dicom_folder(in_dir, out_dir)

print("Done!")
