# merged_fastapi_app.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from db_init import SessionLocal, engine
from db_handling import FaceEncoding, Base
import cv2
import numpy as np
import mediapipe as mp
import base64
from io import BytesIO
from PIL import Image
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import os
import uvicorn

app = FastAPI()

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB Init
Base.metadata.create_all(bind=engine)

# InsightFace model loader
@lru_cache(maxsize=1)
def load_face_model():
    model = FaceAnalysis(name="buffalo_s", root="/tmp/insightface")
    model.prepare(ctx_id=-1)
    return model

# Dependency to get DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Read UploadFile to numpy array
def read_image_from_upload(file: UploadFile):
    image_bytes = file.file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.array(image)

# Extract face embedding from image
def get_face_embedding(file: UploadFile):
    img = read_image_from_upload(file)
    face_app = load_face_model()
    faces = face_app.get(img)
    if not faces:
        raise ValueError("No face found in the image")
    return faces[0].embedding

@app.get("/")
def root():
    return {"message": "Server is running"}

@app.post("/register/")
async def register_user(
    username: str = Form(...),
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        emb1 = get_face_embedding(image1)
        emb2 = get_face_embedding(image2)
        avg_embedding = ((emb1 + emb2) / 2).tolist()

        face_check = db.query(FaceEncoding).filter(FaceEncoding.username == username).first()
        if face_check:
            face_check.encoding = avg_embedding
        else:
            face_check = FaceEncoding(username=username, encoding=avg_embedding)
            db.add(face_check)
        db.commit()

        return {"message": f"{username} registered successfully."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/verify/")
async def verify_user(
    username: str = Form(...),
    live_image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    face_check = db.query(FaceEncoding).filter(FaceEncoding.username == username).first()
    if not face_check:
        return {"verified": "Unknown User", "message": "User not found in our Database"}

    try:
        known_embedding = np.array(face_check.encoding)
        live_embedding = get_face_embedding(live_image)

        similarity = cosine_similarity([known_embedding], [live_embedding])[0][0]
        confidence = similarity * 100
        tolerance = 0.6

        return {
            "verified": "Access granted ✅" if similarity > tolerance else "Access denied ❌",
            "confidence": f"{confidence:.2f}%",
            "message": "User Face matched" if similarity > tolerance else "User Face not matched"
        }
    except Exception as e:
        return {"verified": False, "error": str(e)}

# Live Video WebSocket Verification
@app.websocket("/ws/verify/{username}")
async def verify_live_face(websocket: WebSocket, username: str, db: Session = Depends(get_db)):
    await websocket.accept()

    face_check = db.query(FaceEncoding).filter(FaceEncoding.username == username).first()
    if not face_check:
        await websocket.send_json({
            "verified": False,
            "confidence": 0,
            "message": f"User '{username}' not found in DB"
        })
        await websocket.close()
        return

    known_embedding = np.array(face_check.encoding)
    face_app = load_face_model()

    try:
        while True:
            data = await websocket.receive_text()

            try:
                image_data = base64.b64decode(data.split(",")[-1])
                image = Image.open(BytesIO(image_data)).convert("RGB")
                img = np.array(image)

                faces = face_app.get(img)
                if not faces:
                    await websocket.send_json({
                        "verified": False,
                        "confidence": 0,
                        "message": "No face detected"
                    })
                    continue

                live_embedding = faces[0].embedding
                similarity = cosine_similarity([known_embedding], [live_embedding])[0][0]
                confidence = similarity * 100
                tolerance = 0.6

                await websocket.send_json({
                    "verified": similarity > tolerance,
                    "confidence": f"{confidence:.2f}%",
                    "message": "Face matched ✅" if similarity > tolerance else "Face mismatch ❌"
                })

            except Exception as e:
                await websocket.send_json({"verified": False, "error": str(e)})

    except WebSocketDisconnect:
        print(f"Client {username} disconnected")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)