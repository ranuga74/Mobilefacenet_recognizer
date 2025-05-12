import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from flask import Flask, Response, render_template_string
from train import MobileFaceNet  # Import MobileFaceNet from your train.py

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained MobileFaceNet model
model = MobileFaceNet(embedding_size=512)
model.load_state_dict(torch.load('mobilefacenet_face_recognition1.pth', map_location=device))
model = model.to(device)
model.eval()

# Load embeddings database
embeddings_db = torch.load('mobilefacenet_embeddings1.pth', map_location=device)

# Preprocess transformation for input images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function to compute cosine similarity
def cosine_similarity(emb1, emb2):
    return torch.dot(emb1.flatten(), emb2.flatten()) / (torch.norm(emb1) * torch.norm(emb2))

# Function to find the closest match
def find_match(embedding, embeddings_db, threshold=0.6):
    max_similarity = -1
    matched_person = "Unknown"

    for person_name, embeddings_list in embeddings_db.items():
        for stored_embedding in embeddings_list:
            similarity = cosine_similarity(embedding, stored_embedding.to(device))
            if similarity > max_similarity:
                max_similarity = similarity
                if max_similarity > threshold:
                    matched_person = person_name

    return matched_person, max_similarity.item()

# Flask app
app = Flask(__name__)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract face ROI
            face = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Preprocess face for model
            face_tensor = transform(face_rgb).unsqueeze(0).to(device)

            # Generate embedding
            with torch.no_grad():
                embedding = model(face_tensor)
                embedding = F.normalize(embedding, p=2, dim=1)

            # Find match in embeddings database
            person_name, similarity = find_match(embedding, embeddings_db)

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{person_name} ({similarity:.2f})"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string('''
        <html>
        <body>
            <h1>Live Face Recognition</h1>
            <img src="{{ url_for('video_feed') }}" width="640" height="480">
        </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=5000)
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Live face recognition terminated.")