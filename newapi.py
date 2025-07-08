from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from torchvision import transforms
from PIL import Image
import io
import os
from huggingface_hub import hf_hub_download

from models.TumorModel import TumorClassification, GliomaStageModel
from utils import get_precautions_from_gemini

# ✅ Use Hugging Face's built-in writable cache directory
cache_dir = "/home/user/.cache/huggingface"

# No need to call os.makedirs — directory already exists

# Initialize FastAPI app
app = FastAPI(title="Brain Tumor Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load tumor classification model
btd_model_path = hf_hub_download(
    repo_id="Codewithsalty/brain-tumor-models",
    filename="BTD_model.pth",
    cache_dir=cache_dir
)
tumor_model = TumorClassification()
tumor_model.load_state_dict(torch.load(btd_model_path, map_location="cpu"))
tumor_model.eval()

# Load glioma stage model
glioma_model_path = hf_hub_download(
    repo_id="Codewithsalty/brain-tumor-models",
    filename="glioma_stages.pth",
    cache_dir=cache_dir
)
glioma_model = GliomaStageModel()
glioma_model.load_state_dict(torch.load(glioma_model_path, map_location="cpu"))
glioma_model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

@app.get("/")
async def root():
    return {"message": "Brain Tumor Detection API is running."}

# Labels
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = tumor_model(x)
        idx = torch.argmax(out, dim=1).item()
        tumor_type = labels[idx]

    if tumor_type == "glioma":
        return {"tumor_type": tumor_type, "next": "submit_mutation_data"}
    else:
        precautions = get_precautions_from_gemini(tumor_type)
        return {"tumor_type": tumor_type, "precaution": precautions}

# Mutation input
class MutationInput(BaseModel):
    gender: str
    age: float
    idh1: int
    tp53: int
    atrx: int
    pten: int
    egfr: int
    cic: int
    pik3ca: int

@app.post("/predict-glioma-stage")
async def predict_glioma_stage(data: MutationInput):
    gender_val = 0 if data.gender.lower() == 'm' else 1
    features = [
        gender_val, data.age, data.idh1, data.tp53, data.atrx,
        data.pten, data.egfr, data.cic, data.pik3ca
    ]
    x = torch.tensor(features).float().unsqueeze(0)

    with torch.no_grad():
        out = glioma_model(x)
        idx = torch.argmax(out, dim=1).item()
        stages = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
        return {"glioma_stage": stages[idx]}

# For local development only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("newapi:app", host="0.0.0.0", port=10000)
