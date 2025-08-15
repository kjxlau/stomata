import os
import copy
import random
import json
import yaml
import glob
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
from   zipfile import ZipFile
import argparse
from PIL import Image
import PIL.Image
import shutil
from IPython.display import Image
from sklearn.model_selection import train_test_split
 
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms as T
 
from ultralytics import YOLO

def set_seeds():
    # fix random seeds
    SEED_VALUE = 42
 
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
     
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED_VALUE)
        torch.cuda.manual_seed_all(SEED_VALUE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
set_seeds()

def get_image_mask_pairs(data_dir):
    image_paths = []
    mask_paths = []
 
    for root,_,files in os.walk(data_dir):
        if 'inputs' in root:
            for file in files:
                if file.endswith('.jpg'):
                    image_paths.append(os.path.join(root,file))
                    mask_paths.append(os.path.join(root, file.replace('.jpg','_mask.png')))
    return image_paths, mask_paths
    
def mask_to_polygons(mask,epsilon=1.0):
    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) > 2:
           poly = contour.reshape(-1).tolist()
           if len(poly) > 4: #Ensures valid polygon
              polygons.append(poly)
    return polygons
    
def process_data(image_paths, mask_paths, output_images_dir, output_labels_dir):
    annotations = []
    images = []
    image_id = 0
    ann_id = 0
 
    for img_path, mask_path in zip(image_paths, mask_paths):
        image_id += 1
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        shutil.copy(img_path, os.path.join(output_images_dir, os.path.basename(img_path)))
 
        # Add image to the list
        images.append({
            "id": image_id,
            "file_name": os.path.basename(img_path),
            "height": img.shape[0],
            "width": img.shape[1]
        })
 
        unique_values = np.unique(mask)
        for value in unique_values:
            if value == 0:  # Ignore background
                continue
 
            object_mask = (mask == value).astype(np.uint8) * 255
            polygons = mask_to_polygons(object_mask)
 
            for poly in polygons:
                ann_id += 1
                annotations.append({
                   
                    "image_id": image_id,
                    "category_id": 1,  # Only one category: Nuclei
                    "segmentation": [poly],  
                })
                coco_input = {
       "images": images,
       "annotations": annotations,
       "categories": [{"id": 1, "name": "stomata"}]
   }
                for img_info in coco_input["images"]:
                    img_id = img_info["id"]
                    img_ann = [ann for ann in coco_input["annotations"] if ann["image_id"] == img_id]
                    img_w, img_h = img_info["width"], img_info["height"]
                    if img_ann:
                        with open(os.path.join(output_labels_dir, os.path.splitext(img_info["file_name"])[0] + '.txt'), 'w') as file_object:
                            for ann in img_ann:
                                current_category = ann['category_id'] - 1
                                polygon = ann['segmentation'][0]
                                normalized_polygon = [format(coord / img_w if i % 2 == 0 else coord / img_h, '.6f') for i, coord in enumerate(polygon)]
                                file_object.write(f"{current_category} " + " ".join(normalized_polygon) + "\n")
                                
def create_yaml(output_yaml_path, train_images_dir, val_images_dir, nc=1):
    # Assuming all categories are the same and there is only one class, 'Nuclei'
    names = ['stomata']
 
    # Create a dictionary with the required content
    yaml_data = {
        'names': names,
        'nc': nc,  # Number of classes
        'train': train_images_dir,
        'val': val_images_dir,
        'test': ' '
    }
    # Write the dictionary to a YAML file
    with open(output_yaml_path, 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)

def yolo_dataset_preparation():
    data_dir = './'
    output_dir = 'yolov12s_dataset'
 
    # Define the paths for the images and labels for training and validation
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    val_labels_dir = os.path.join(output_dir, 'val', 'labels')
 
    # Create the output directories if they do not exist
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
 
    # Get image and mask paths
    image_paths, mask_paths = get_image_mask_pairs(data_dir)
 
    # Split data into train and val
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)
 
    # Process and save the data in YOLO format for training and validation
    process_data(train_img_paths, train_mask_paths, train_images_dir, train_labels_dir)
    process_data(val_img_paths, val_mask_paths, val_images_dir, val_labels_dir)
    
    # Assume create_yaml function is defined elsewhere and set appropriate paths for the YAML file
    output_yaml_path = os.path.join(output_dir, 'data.yaml')
    train_path = os.path.join('train', 'images')
    val_path = os.path.join('val', 'images')
    create_yaml(output_yaml_path, train_path, val_path)
 
yolo_dataset_preparation()

model = YOLO("yolov12s.yaml") #build a model from YAML

with open("yolov12s_dataset/data.yaml",'r') as stream:
     num_classes = str(yaml.safe_load(stream)['nc'])
     
#Define a project --> Destination directory for all results
project = "yolov12s_dataset/results"
#Define subdirectory for this specific training
name = "300_epochs-v12s"
ABS_PATH = os.getcwd()

#Train the model
results = model.train(
    data = os.path.join(ABS_PATH, "yolov12s_dataset/data.yaml"),
    project = project,
    name = name,
    epochs = 300,
    mosaic = 1.0,
    scale = 0.5,
    patience = 0 , #setting patience=0 to disable early stopping,
    flipup = 0.5,
    fliplr = 0.5,
    degrees = 90.0,
    batch = 3,
    imgsz=1200
)
