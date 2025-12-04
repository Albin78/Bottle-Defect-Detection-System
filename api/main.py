from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import shutil
import os
import uuid
from ai.predict import detect_defect

app = FastAPI()

# --- Pydantic Models ---

class Detection(BaseModel):
    class_name: str = Field(..., alias="class")
    class_id: int
    confidence: float

class PredictionResponse(BaseModel):
    filename: str
    is_defect: bool
    label: str
    detections: Optional[List[Detection]] = None
    processed_image_path: str

# --- Endpoints ---

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save uploaded file to a temporary location
    temp_filename = f"temp_{uuid.uuid4().hex}_{file.filename}"
    temp_file_path = os.path.join("temp_uploads", temp_filename)
    os.makedirs("temp_uploads", exist_ok=True)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Call the detection function
        # detect_defect expects a file path
        result = detect_defect(temp_file_path, save_output=True)
        
        # Construct response
        # Map dictionary keys to Pydantic model
        response = PredictionResponse(
            filename=file.filename,
            is_defect=result["is_defect"],
            label=result["label"],
            detections=result["detections"],
            processed_image_path=result["image_path"]
        )
        
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up the temporary uploaded file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/")
def read_root():
    return {"message": "Bottle Defect Detection API is running"}
