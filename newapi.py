from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from torchvision import transforms
from PIL import Image
import io
import os

# ✅ Set Hugging Face model cache directory to a writable path
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"

from huggingface_hub import hf_hub_download
from models.TumorModel import TumorClassification, GliomaStageModel
from utils import get_precautions_from_gemini

# Define your app
app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load your models from the Hugging Face Hub
btd_model_path = hf_hub_download(repo_id="Codewithsalty/brain-tumor-detection", filename="brain_tumor_model.pt")
glioma_model_path = hf_hub_download(repo_id="Codewithsalty/brain-tumor-detection", filename="glioma_stage_model.pt")

btd_model = TumorClassification(model_path=btd_model_path)
glioma_model = GliomaStageModel(model_path=glioma_model_path)

# ✅ Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class DiagnosisResponse(BaseModel):
    tumor: str
    stage: str
    precautions: list

@app.post("/predict", response_model=DiagnosisResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    tumor_result = btd_model.predict(image_tensor)
    if tumor_result == "No Tumor":
        return DiagnosisResponse(
            tumor="No Tumor Detected",
            stage="N/A",
            precautions=[]
        )

    stage_result = glioma_model.predict(image_tensor)
    precautions = get_precautions_from_gemini(tumor_result, stage_result)

    return DiagnosisResponse(
        tumor=tumor_result,
        stage=stage_result,
        precautions=precautions
    )

@app.get("/")
def root():
    return {"message": "Brain Tumor API is running."}
