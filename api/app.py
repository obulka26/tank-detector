from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from model.src.predict import predict_single_image

app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    result = predict_single_image(image)
    return {"filename": file.filename, "prediction": result}
