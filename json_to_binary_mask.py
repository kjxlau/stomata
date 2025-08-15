import json
import numpy as np
import cv2
import os
import glob

def json_to_binary_mask(json_file_path, output_path):
    """
    Converts a LabelMe JSON file to a binary mask.

    Args:
        json_file_path (str): The path to the LabelMe JSON file.
        output_path (str): The path to save the binary mask.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Get the image dimensions
    height = data['imageHeight']
    width = data['imageWidth']

    # Create a blank mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Iterate through the shapes and draw them on the mask
    for shape in data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

    # Save the mask
    cv2.imwrite(output_path, mask)

if __name__ == '__main__':
    # --- Main processing loop ---

    # 1. Define the directory containing your JSON files.
    json_directory = './'

    # 2. Define the directory where you want to save the output masks.
    #    The script will create this directory if it doesn't exist.
    output_directory = './masks'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 3. Find all files ending with .json in the specified directory.
    json_files = glob.glob(os.path.join(json_directory, '*.json'))

    # 4. Loop through each JSON file found.
    for json_file in json_files:
        # Create a corresponding output filename for the mask.
        # It takes the base name of the JSON file (e.g., "Rice_40x_320")
        # and creates a new path in the output directory.
        base_name = os.path.basename(json_file) # e.g., "Rice_40x_320.json"
        file_name_without_ext = os.path.splitext(base_name)[0] # e.g., "Rice_40x_320"
        output_file = os.path.join(output_directory, f"{file_name_without_ext}_mask.png")

        # Process the file and generate the mask
        print(f"Processing '{json_file}' -> Saving mask to '{output_file}'")
        json_to_binary_mask(json_file, output_file)
