import cv2
import torch
from torchvision import transforms
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np

from models.cnn_models import EmotionCNN
from utils.utils import get_transforms

# Khởi tạo FastAPI và cấu hình thư mục static, templates
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Cấu hình thiết bị tính toán
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, test_transform = get_transforms()

# Load model và trọng số
model = EmotionCNN(num_classes=7).to(device)
model.load_state_dict(torch.load("models/model_weights.pth", map_location=device))
model.eval()

# Nhãn cảm xúc
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load Haar cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Hàm dự đoán cảm xúc từ ảnh khuôn mặt
def predict_emotion(face_img):
    # Áp dụng CLAHE để tăng cường độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    face_img = clahe.apply(face_img)
    
    # Chuyển ảnh về định dạng PIL và áp dụng transform
    face_pil = transforms.ToPILImage()(face_img)
    input_tensor = test_transform(face_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        preds = outputs.argmax(dim=1)
        emotion = emotion_labels[preds.item()]
    return emotion

# Generator để phát video stream
def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Không thể mở camera.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Chuyển frame sang grayscale để phát hiện khuôn mặt
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                try:
                    # Resize face về kích thước 48x48 như training
                    face_resized = cv2.resize(face, (48, 48))
                    emotion = predict_emotion(face_resized)
                except Exception as e:
                    emotion = "error"
                # Vẽ khung và nhãn lên frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 200, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2, cv2.LINE_AA)

            # Mã hóa frame thành JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

# Trang chủ hiển thị giao diện web
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint cung cấp MJPEG stream từ webcam
@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
