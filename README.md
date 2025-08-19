# Stomata Detection
A dataset comprising 1,000 microscopy images was gathered from the abaxial surfaces of barley and rice leaves. Barley leaf images were captured at 200x magnification, while rice leaf images were taken at 400x. Stomata within these images were manually annotated using the LabelMe polygon tool, and the resulting JSON annotations were transformed into binary mask formats. These annotated images and corresponding masks served as training data for the YOLOv12 object detection model. Dataset were split into 80% training and 20% for test validation. 

## 1. Manual Annotation of Images with LabelMe
Download and install app from https://github.com/wkentaro/labelme
<br> Using the polygon tool, draw a polygon object around objects of interest and label it with a class name "stomata". </br>
A JSON annotation file will be created of the class object and coordinate space in the corresponding image when you hit the 'save' button.

<img width="622" height="464" alt="labelme" src="https://github.com/user-attachments/assets/32442cec-06da-446b-a9e8-9fcefe2cdec0" />

## 2. JSON to Binary Mask script
Run python json_to_binary_mask.py 
<br> To convert JSON coordinate index files into binary mask in PNG format </br>

## 3. Convert segmented binary mask into YOLO format and initiate YOLOv12 model training
Run python train_yolo_model.py
<br> Specify the number of epoch cycles and augmentation </br>
Augmentation includes scaling the image up, flipping left-right, flipping up-down and rotating at various angles to train the model to be robust.
<br> https://rumn.medium.com/yolo-data-augmentation-explained-turbocharge-your-object-detection-model-94c33278303a </br>

<img width="403" height="272" alt="image" src="https://github.com/user-attachments/assets/c81b17e0-f4c4-416b-ab30-4e2ec2e39638" />

YOLOv12 integrates Flash Attention to dramatically reduce computation time during training and inference. This optimization is especially effective on modern GPU architectures, including Turing: NVIDIA T4, Quadro RTX series, Ampere: RTX 30 series, A30, A40, A100, Ada Lovelace: RTX 40 series, Hopper: H100, H200. While GPU acceleration is highly recommended for optimal performance, YOLOv12 remains compatible with CPU-based training. However, users should expect significantly longer runtimes when operating without GPU support.

<img width="1346" height="814" alt="image" src="https://github.com/user-attachments/assets/67dd1075-3305-4364-9a0b-9e0898933e9f" />

## 4. Image analysis and stomatal traits inference
<img width="1667" height="929" alt="image" src="https://github.com/user-attachments/assets/a2940670-67a4-474c-8138-48c1aafa27c5" />
