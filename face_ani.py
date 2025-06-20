from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
import base64
from io import BytesIO
from PIL import Image
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

@app.websocket("/ws/face")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()

            try:
                image_data = base64.b64decode(data.split(",")[-1])
                image = Image.open(BytesIO(image_data)).convert("RGB")
                frame = np.array(image)
                frame = cv2.flip(frame, 1)
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                results = face_detection.process(bgr)

                if results.detections:
                    ih, iw, _ = frame.shape
                    largest_face = None
                    max_area = 0

                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        x = int(bboxC.xmin * iw)
                        y = int(bboxC.ymin * ih)
                        w = int(bboxC.width * iw)
                        h = int(bboxC.height * ih)
                        area = w * h

                        if area > max_area:
                            max_area = area
                            largest_face = {"x": x, "y": y, "w": w, "h": h}

                    await websocket.send_json({"success": True, "face": largest_face})
                else:
                    await websocket.send_json({"success": True, "face": None})

            except Exception as e:
                await websocket.send_json({"success": False, "error": str(e)})
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)