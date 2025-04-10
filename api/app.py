from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import io
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

    predictions, image_path = predict_single_image(image)

    return JSONResponse(
        content={
            "filename": file.filename,
            "predictions": predictions,
            "image_url": "/result",
        }
    )


@app.get("/result")
async def get_result():
    return FileResponse("result.jpg", media_type="image/jpeg")
