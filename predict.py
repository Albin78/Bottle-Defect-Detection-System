import cv2
from ultralytics import YOLO

# Load the model
model = YOLO("runs/detect/train4/weights/best.pt")

# Define defect class indices based on data.yaml
# names: ['bottle', 'cap', 'cap missing', 'damaged plastic', 'label', 'label missing']
DEFECT_CLASSES = [2, 3, 5] 

# Run prediction
# stream=True is recommended for processing multiple images or video to manage memory
results = model.predict("images/non-defect/", 
                        save=True,
                        conf=0.5,
                        imgsz=640,
                        stream=True)

for result in results:
    # Check if any detected object is a defect
    is_defect = False
    if result.boxes:
        for cls in result.boxes.cls:
            if int(cls) in DEFECT_CLASSES:
                is_defect = True
                print("Defect detected:", int(cls))
                break
    
    # Determine label and color
    label = "Defect" if is_defect else "Perfect"
    color = (0, 0, 255) if is_defect else (0, 255, 0) # Red for Defect, Green for Non-Defect
    
    # Get the plotted image (numpy array)
    img = result.plot()
    
    # Add the global label to the top-left corner
    # Reduced font scale from 1.5 to 1.0 and thickness from 3 to 2
    cv2.putText(img, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    # Create a resizable window
    cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
    # Display the image
    cv2.imshow("Prediction", img)
    
    # Wait for a key press to move to the next image (or exit)
    # 0 waits indefinitely, 1 waits 1ms. Using 0 so user can see each result.
    key = cv2.waitKey(0) 
    if key == ord('q'):
        break

cv2.destroyAllWindows()

