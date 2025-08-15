import json
import numpy as np
import cv2

def labelme_to_binary_mask(json_file_path, output_path):
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
    # Example usage
    json_file = 'Rice_40x_320.json'
    output_file = './Rice_40x_320_mask.png'
    labelme_to_binary_mask(json_file, output_file)
