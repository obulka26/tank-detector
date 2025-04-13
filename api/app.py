from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import cv2
import tempfile
import os
import uuid
import shutil
import imageio
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


def extract_frames(video_path, sample_count=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indexes = [int(i * frame_count / sample_count) for i in range(sample_count)]

    frames = []
    current_idx = 0
    success, frame = cap.read()
    while success:
        if current_idx in frame_indexes:
            frames.append((current_idx, frame))
        success, frame = cap.read()
        current_idx += 1
    cap.release()
    return frames


def draw_bboxes_on_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    success, frame = cap.read()
    while success:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, processed = predict_single_image(rgb)
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        out.write(processed_bgr)
        success, frame = cap.read()

    cap.release()
    out.release()


def video_to_gif(video_path, gif_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        success, frame = cap.read()
    imageio.mimsave(gif_path, frames, duration=0.1, loop=0)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    if file.filename.endswith((".jpg", ".jpeg", ".png")):
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
    elif file.filename.endswith((".mp4", ".avi", ".mov")):
        suffix = os.path.splitext(file.filename)[-1]
        temp_dir = tempfile.mkdtemp()
        raw_video_path = os.path.join(temp_dir, f"{uuid.uuid4()}{suffix}")
        with open(raw_video_path, "wb") as f:
            f.write(contents)

        has_tank = False
        sample_frames = extract_frames(raw_video_path, sample_count=8)
        for _, frame in sample_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictions, _ = predict_single_image(rgb)
            if any(pred["label"] == "tank" for pred in predictions):
                has_tank = True
                break

        if has_tank:
            processed_video_path = os.path.join(
                temp_dir, f"processed_{uuid.uuid4()}.mp4"
            )
            draw_bboxes_on_video(raw_video_path, processed_video_path)
        else:
            processed_video_path = raw_video_path

        input_gif = os.path.join(temp_dir, "input.gif")
        output_gif = os.path.join(temp_dir, "output.gif")

        video_to_gif(raw_video_path, input_gif)
        video_to_gif(processed_video_path, output_gif)

        with open(input_gif, "rb") as f:
            input_gif_bytes = f.read()
        with open(output_gif, "rb") as f:
            output_gif_bytes = f.read()

        shutil.rmtree(temp_dir)

        return JSONResponse(
            content={
                "has_tank": has_tank,
                "input_gif": base64.b64encode(input_gif_bytes).decode(),
                "output_gif": base64.b64encode(output_gif_bytes).decode(),
            }
        )
    else:
        return {"error": "Unsupported file format"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
