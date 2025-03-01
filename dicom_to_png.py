import pydicom
import numpy as np
from PIL import Image
import os
import shutil
import argparse
import pandas as pd

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
    
    # black out info
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

def extract_lesions(dicom_path, annotations_df, output_root, target_size=(512, 512), apply_resize=True):
    """
    Extract lesion regions from a DICOM image based on bounding boxes provided in annotations_df.
    
    The CSV (finding_annotations.csv) must include at least the following columns:
      image_id, study_id, xmin, ymin, xmax, ymax
    Each lesion is saved as a separate PNG in output_root/<study_id>/lesions/.
    """
    try:
        dicom_image = pydicom.dcmread(dicom_path)
        if not hasattr(dicom_image, "pixel_array"):
            print(f"Skipping {dicom_path}: No pixel data found.")
            return

        # normalization
        image_array = dicom_image.pixel_array
        image_array = (image_array - np.min(image_array)) / (np.ptp(image_array)) * 255.0
        image_array = image_array.astype(np.uint8)

        # get image ID from the filename
        image_id = os.path.splitext(os.path.basename(dicom_path))[0]
        # get annotations for respective ID
        lesion_rows = annotations_df[annotations_df['image_id'] == image_id]

        # process each single lesion
        for idx, row in lesion_rows.iterrows():
            # handle images without lesions
            if pd.isnull(row[['xmin', 'ymin', 'xmax', 'ymax']]).any():
                continue 
        
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])

            # extract lesions using annotation bounding boxes
            lesion_region = image_array[ymin:ymax, xmin:xmax]
            lesion_img = Image.fromarray(lesion_region)

            if apply_resize:
                lesion_img = resize_with_padding(lesion_img, target_size=target_size)

            # grab study_id from the CSV for output folder
            #study_id = str(row['study_id'])
            #lesion_out_dir = os.path.join(output_root, study_id, "lesions")
            #os.makedirs(lesion_out_dir, exist_ok=True)
            lesion_filename = f"{image_id}_lesion_{idx}.png"
            #lesion_out_path = os.path.join(lesion_out_dir, lesion_filename)
            lesion_out_path = os.path.join(output_root, lesion_filename)
            lesion_img.save(lesion_out_path)
            print(f"Extracted lesion: {lesion_out_path}")

    # catch exceptions
    except Exception as e:
        print(f"Error extracting lesions from {dicom_path}: {e}")

def process_dicom_folder(input_root, output_root, target_size=(512, 512), apply_resize=False, lesions_flag=False, annotations_df=None):
    """
    Process all DICOM images in subfolders, converting them to PNG.
    
    If apply_resize is True, each image is resized (with padding) to target_size.
    If lesions_flag is False, converts full DICOM images to PNG (cropped/resized as specified).
    If True, extracts lesion regions based on bounding boxes from annotations_df.

    Also copies index.html files.
    """
    for subdir, _, files in os.walk(input_root):
        for file in files:
            input_file_path = os.path.join(subdir, file)
            if file.lower().endswith(".dicom"):
                if lesions_flag:
                    extract_lesions(input_file_path, annotations_df, output_root, target_size=target_size, apply_resize=apply_resize)
                else:        
                    relative_path = os.path.relpath(subdir, input_root)
                    output_subdir = os.path.join(output_root, relative_path)
                    os.makedirs(output_subdir, exist_ok=True)
                    output_file_path = os.path.join(output_subdir, os.path.splitext(file)[0] + ".png")
                    dicom_to_png(input_file_path, output_file_path, target_size=target_size, apply_resize=apply_resize)
            elif file == "index.html" and not lesions_flag:
                relative_path = os.path.relpath(subdir, input_root)
                output_subdir = os.path.join(output_root, relative_path)
                shutil.copy(input_file_path, output_subdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert DICOM images to PNG with cropping/resizing."
    )
    parser.add_argument("in_folder", help="Path to the input folder containing DICOM images.")
    parser.add_argument("out_folder", help="Path to the output folder for PNG images.")
    parser.add_argument("--resize", action="store_true", 
                        help="Apply resizing with padding to a uniform target size.")
    parser.add_argument("--lesions", action="store_true", 
                        help="Extract lesions based on finding_annotations.csv.")
    
    args = parser.parse_args()

    # load annotation CSV for lesion extraction
    annotations_df = None
    if args.lesions:
        try:
            annotations_df = pd.read_csv("../shared_data/VinDr_Mammo/finding_annotations.csv") # NEED TO KEEP THIS UP TO DATE
            print("Loaded finding_annotations.csv")
        except Exception as e:
            print(f"Error loading finding_annotations.csv: {e}")
            exit(1)

    process_dicom_folder(args.in_folder, args.out_folder,
                         apply_resize=args.resize,
                         lesions_flag=args.lesions,
                         annotations_df=annotations_df)

    print("Done!")
