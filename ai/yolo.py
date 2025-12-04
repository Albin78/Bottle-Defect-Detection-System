import os
import zipfile
from ultralytics import YOLO


zip_path = "Bottle Defect Detection.v1i.yolov8.zip"
extract_path = "extracted/"
extracted = False

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Fix data.yaml paths
yaml_path = os.path.join(extract_path, "data.yaml")
if os.path.exists(yaml_path):
    with open(yaml_path, 'r') as f:
        content = f.read()
    
    # Replace incorrect relative paths
    new_content = content.replace('../train/images', 'train/images')
    new_content = new_content.replace('../valid/images', 'valid/images')
    new_content = new_content.replace('../test/images', 'test/images')
    
    with open(yaml_path, 'w') as f:
        f.write(new_content)
    print("Fixed paths in data.yaml")

    print("Extraction completed.")
    model = YOLO("yolov8n.pt")
    data_path = "data.yaml"

    model.train(
        data = os.path.join(extract_path, data_path),
        epochs = 10,
        imgsz = 640,        
    )
else:
    print("Extraction failed.")


