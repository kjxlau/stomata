import os
import json
import argparse
import numpy as np
from PIL import Image

def get_class_names(class_file):
    """
    Reads class names from a file.
    """
    with open(class_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def yolo_segmentation_to_coco(image_dir, label_dir, class_names):
    """
    Converts YOLO formatted segmentation annotations to COCO format.
    """
    coco_output = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Create categories
    for i, class_name in enumerate(class_names):
        coco_output["categories"].append({
            "id": i,
            "name": class_name,
            "supercategory": "object",
        })

    image_id_counter = 0
    annotation_id_counter = 0

    for filename in os.listdir(image_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_dir, filename)
            
            # Get image dimensions
            with Image.open(image_path) as img:
                width, height = img.size

            # Add image info
            image_info = {
                "id": image_id_counter,
                "file_name": filename,
                "width": width,
                "height": height,
            }
            coco_output["images"].append(image_info)

            # Corresponding label file
            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(label_dir, label_filename)

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) > 1:
                            class_id = int(float(parts[0]))
                            
                            # The rest are normalized polygon points
                            poly_norm = [float(p) for p in parts[1:]]
                            
                            # De-normalize polygon points
                            poly_denorm = []
                            for i in range(0, len(poly_norm), 2):
                                x = poly_norm[i] * width
                                y = poly_norm[i+1] * height
                                poly_denorm.extend([x, y])

                            # Calculate bounding box from polygon
                            poly_np = np.array(poly_denorm).reshape(-1, 2)
                            x_min, y_min = np.min(poly_np, axis=0)
                            x_max, y_max = np.max(poly_np, axis=0)
                            bbox_width = x_max - x_min
                            bbox_height = y_max - y_min
                            
                            # Area can be bbox area for simplicity
                            area = bbox_width * bbox_height

                            annotation_info = {
                                "id": annotation_id_counter,
                                "image_id": image_id_counter,
                                "category_id": class_id,
                                "segmentation": [poly_denorm],
                                "area": area,
                                "bbox": [x_min, y_min, bbox_width, bbox_height],
                                "iscrowd": 0,
                            }
                            coco_output["annotations"].append(annotation_info)
                            annotation_id_counter += 1
            
            image_id_counter += 1

    return coco_output

def main():
    parser = argparse.ArgumentParser(description="Convert YOLO segmentation annotations to COCO format.")
    parser.add_argument("image_dir", type=str, help="Path to the directory containing images.")
    parser.add_argument("label_dir", type=str, help="Path to the directory containing YOLO annotation files.")
    parser.add_argument("class_file", type=str, help="Path to the file containing class names.")
    parser.add_argument("output_file", type=str, help="Path to save the output COCO JSON file.")
    args = parser.parse_args()

    class_names = get_class_names(args.class_file)
    coco_data = yolo_segmentation_to_coco(args.image_dir, args.label_dir, class_names)

    with open(args.output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f"Successfully converted YOLO segmentation annotations to COCO format at: {args.output_file}")

if __name__ == "__main__":
    main()
