# Stomata Detection
A dataset comprising 1,000 microscopy images was gathered from the abaxial surfaces of barley and rice leaves. Barley leaf images were captured at 200x magnification, while rice leaf images were taken at 400x. Stomata within these images were manually annotated using the LabelMe polygon tool, and the resulting JSON annotations were transformed into binary mask formats. 
<br> These annotated images and corresponding masks served as training data for the YOLOv12 object detection model. Dataset were split into 80% training and 20% for test validation. </br>

## 1. Manual Annotation of Images with LabelMe
Please install app from https://github.com/wkentaro/labelme
<br> Draw polygon boxes around object of interest and label it with a class name "stomata". </br>
A JSON annotation file will be created of the class object and coordinate space in the corresponding image when you hit the 'save' button.

<img width="622" height="464" alt="labelme" src="https://github.com/user-attachments/assets/32442cec-06da-446b-a9e8-9fcefe2cdec0" />

## 2. JSON to Binary Mask script
Run python json_to_binary_mask.py to convert JSON coordinate index files into binary mask in PNG format

<img width="1667" height="929" alt="image" src="https://github.com/user-attachments/assets/a2940670-67a4-474c-8138-48c1aafa27c5" />
