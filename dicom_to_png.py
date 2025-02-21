import pydicom
import numpy as np
from PIL import Image
import os

def dicom_to_png(dicom_path, output_path):
    # Load the DICOM file
    dicom_image = pydicom.dcmread(dicom_path)
    
    # Convert pixel data to a NumPy array
    image_array = dicom_image.pixel_array
    
    # Normalize the image (optional, improves visibility)
    image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255.0
    image_array = image_array.astype(np.uint8)

    # Convert to PIL Image and save as PNG
    image = Image.fromarray(image_array)
    image.save(output_path)

# Example usage
dicom_file = "example.dcm"
output_file = "output.png"
dicom_to_png(dicom_file, output_file)


input_folder = "images"
output_folder = "images_png"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".dcm"):
        dicom_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file.replace(".dcm", ".png"))
        dicom_to_png(dicom_path, output_path)

print("Conversion complete!")
