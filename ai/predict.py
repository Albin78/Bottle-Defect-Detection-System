import cv2
import uuid
import os
from ultralytics import YOLO

# Load the model
# Ensure the model path is correct relative to where this script is run or the FastAPI app is run
model = YOLO("runs/detect/train4/weights/best.pt")

# Define defect class indices based on data.yaml
# names: ['bottle', 'cap', 'cap missing', 'damaged plastic', 'label', 'label missing']
DEFECT_CLASSES = [2, 3, 5] 

def detect_defect(image_path: str, save_output: bool = True):
    """
    Detects defects in an image using the loaded YOLO model.

    Args:
        image_path (str): Path to the image file.
        save_output (bool): Whether to save the annotated image.

    Returns:
        dict: A dictionary containing detection results and the path to the processed image.
    """
    # Run prediction
    # stream=False (default) returns a list of Results objects
    results = model.predict(image_path, 
                            save=save_output,
                            conf=0.25, # Lowered confidence to catch more detections
                            imgsz=640,
                            stream=False 
            )
    
    result = results[0]
    detections = []
    is_defect = False

    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = result.names[cls]

        detections.append({
            "class": class_name,
            "class_id": cls,
            "confidence": round(conf, 3)
        })

        if cls in DEFECT_CLASSES:
            is_defect = True
            
    # Determine label
    label = "Defect" if is_defect else "Perfect"
    
    output_path = ""
    if save_output:
        annotated = result.plot()
        filename = f"{uuid.uuid4().hex}.jpg"
        output_folder = "results"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, annotated)

    return {
        "is_defect": is_defect,
        "label": label,
        "detections": detections if detections else None,
        "image_path": output_path
    }


if __name__ == "__main__":
    # Test the function
    test_image_path = "images/defect/d3.jpg"
    if os.path.exists(test_image_path):
        result = detect_defect(test_image_path)
        print(result)
    else:
        print(f"Test image not found: {test_image_path}")

