import pydicom
import numpy as np
from PIL import Image
import os
import shutil
import argparse

def crop_borders(image_array, threshold=10):
    """
    Crop the image to the smallest rectangle containing all pixels above a threshold,
    ignoring corners containing patient informartion according to documentation.
    
    Parameters:
    - image_array: The normalized image array.
    - threshold: Minimum intensity to be considered non-black.
    
    Returns:
    - Cropped image array.
    """
    white_threshold = 150
    
    # mask out info
    rem_info = image_array.copy()
    h, w = rem_info.shape
    rem_info[:45, :80] = 0       # top left
    rem_info[:45, w-80:w] = 0    # top right
    rem_info[h-45:h, :80] = 0    # bottom left
    rem_info[h-45:h, w-80:w] = 0 # bottom right
    rem_info[rem_info > white_threshold] = 0

    # binary mask for above and below threshold
    mask = rem_info > threshold
    
    # catch problems with empty mask
    if not mask.any():
        return image_array

    # collect coordinates of the non-black pixels
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    
    # crop to these boundaries
    #return image_array[y0:y1+1, x0:x1+1]
    return image_array[y0:y1+1, x0:x1+1]

def resize_with_padding(image, target_size=(512, 512)):
    """
    Resize image and pad to target_size.
    
    Parameters:
    - image: A PIL Image.
    - target_size: Tuple (width, height) for desired output size.
    
    Returns:
    - A new padded PIL Image of size target_size.
    """
    original_size = image.size  # (width, height)
    target_width, target_height = target_size

    # get scale factor and new size to fit
    scale = min(target_width / original_size[0], target_height / original_size[1])
    new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    
    # resize
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

    # create image with desired size
    new_image = Image.new("L", target_size, 0)
    
    # add cropped image to padding "passpartout"
    paste_position = ((target_width - new_size[0]) // 2, (target_height - new_size[1]) // 2)
    new_image.paste(resized_image, paste_position)
    
    return new_image

def dicom_to_png(dicom_path, output_path, target_size=(512, 512), apply_resize=True):
    """
    Convert DICOM image to PNG and crop it to a desired target size.
    
    If apply_resize is True, the cropped image is padded to fit the target_size.
    Otherwise, the cropped image is saved as is.
    """
    try:
        dicom_image = pydicom.dcmread(dicom_path)

        # handle problematic images
        if not hasattr(dicom_image, "pixel_array"):
            print(f"Skipping {dicom_path}: No pixel data found.")
            return

        image_array = dicom_image.pixel_array

        # normalize pixel values to 0-255
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255.0
        image_array = image_array.astype(np.uint8)

        # crop the image 
        cropped_array = crop_borders(image_array)

        # convert to a PIL
        image = Image.fromarray(cropped_array)

        # if desired, resize with black padding to target size
        if apply_resize:
            image = resize_with_padding(image, target_size=target_size)

        # save the output PNG
        image.save(output_path)
        print(f"Processed: {dicom_path} → {output_path}")

    # catch exceptions
    except Exception as e:
        print(f"Error processing {dicom_path}: {e}")

def process_dicom_folder(input_root, output_root, target_size=(512, 512), apply_resize=False):
    """
    Process all DICOM images in subfolders, converting them to PNG.
    
    If apply_resize is True, each image is resized (with padding) to target_size.
    Also copies index.html files.
    """
    for subdir, _, files in os.walk(input_root):
        relative_path = os.path.relpath(subdir, input_root)
        output_subdir = os.path.join(output_root, relative_path)
        os.makedirs(output_subdir, exist_ok=True)

        for file in files:
            input_file_path = os.path.join(subdir, file)

            # Process DICOM files (case-insensitive)
            if file.lower().endswith(".dicom"):
                output_file_path = os.path.join(output_subdir, os.path.splitext(file)[0] + ".png")
                dicom_to_png(input_file_path, output_file_path, target_size=target_size, apply_resize=apply_resize)
            elif file == "index.html":
                shutil.copy(input_file_path, output_subdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert DICOM images to PNG with cropping/resizing."
    )
    parser.add_argument("in_folder", help="Path to the input folder containing DICOM images.")
    parser.add_argument("out_folder", help="Path to the output folder for PNG images.")
    parser.add_argument("--resize", action="store_true", 
                        help="Apply resizing with padding to a uniform target size.")
    
    args = parser.parse_args()
    
    process_dicom_folder(args.in_folder, args.out_folder,
                         apply_resize=args.resize)
    
    print("Done!")
