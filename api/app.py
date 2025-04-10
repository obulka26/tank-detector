from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
from fastapi.middleware.cors import CORSMiddleware
from model.object_detection_model.src.predict import predict_single_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    predictions, processed_image = predict_single_image(image)

    pil_image = Image.fromarray(processed_image)

    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    buffered.seek(0)
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return JSONResponse(
        content={
            "predictions": predictions,
            "processed_image": encoded_image,
        }
    )
