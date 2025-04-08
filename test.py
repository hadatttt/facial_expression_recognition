import torch
from torchvision import transforms
from models.cnn_models import EmotionCNN
from utils.utils import get_transforms
import cv2
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load transforms
_, test_transform = get_transforms()

# Load model
model = EmotionCNN(num_classes=7).to(device)
model.load_state_dict(torch.load('models/model_weights.pth', map_location=device))
model.eval()

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Predict emotion function (improved)
def predict_emotion(face_img):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    face_img = clahe.apply(face_img)

    # Resize and apply test_transform
    face_pil = transforms.ToPILImage()(face_img)
    input_tensor = test_transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        preds = outputs.argmax(dim=1)
        emotion = emotion_labels[preds.item()]
    return emotion  

# Start camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))

            # Predict emotion for face
            emotion = predict_emotion(face)

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 200, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Emotion Recognition", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
