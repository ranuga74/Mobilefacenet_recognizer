import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from flask import Flask, Response, render_template_string, request, jsonify
from model import MobileFaceNet  # Import MobileFaceNet from your train.py
import base64
import logging
from dotenv import load_dotenv
import os
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()
UPLOAD_FOLDER = '/tmp/Uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Setup logging
logging.basicConfig(filename='/tmp/app.log', level=logging.INFO
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained MobileFaceNet model
try:
    model = MobileFaceNet(embedding_size=512)
    model.load_state_dict(torch.load('mobilefacenet_face_recognition1.pth', map_location=device))
    model = model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    exit()

# Load embeddings database
try:
    embeddings_db = torch.load('mobilefacenet_embeddings1.pth', map_location=device)
    logger.info("Embeddings database loaded successfully")
except Exception as e:
    logger.error(f"Error loading embeddings database: {e}")
    exit()

# Preprocess transformation for input images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    logger.error("Error: Could not load Haar cascade classifier.")
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

# Function to process a single image for enrollment
def process_image_for_enrollment(img, person_name):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) != 1:
        return False, "Image must contain exactly one face."
    
    face_tensor = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(face_tensor)
        embedding = F.normalize(embedding, p=2, dim=1)
    
    if person_name not in embeddings_db:
        embeddings_db[person_name] = []
    embeddings_db[person_name].append(embedding.cpu())
    return True, "Success"

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template_string('''
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {
                    margin: 0;
                    font-family: Arial, sans-serif;
                    background: linear-gradient(to bottom, #e0e7ff, #ffffff);
                    text-align: center;
                    padding: 20px;
                }
                h1 {
                    font-size: 2em;
                    color: #1e3a8a;
                    margin-bottom: 20px;
                }
                .container {
                    max-width: 1000px;
                    margin: 0 auto;
                    background: #ffffff;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                }
                canvas {
                    max-width: 100%;
                    height: auto;
                    border: 3px solid #1e3a8a;
                    border-radius: 8px;
                    margin: 10px 0;
                }
                .button {
                    padding: 12px 24px;
                    margin: 10px;
                    background: #1e3a8a;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 1em;
                    transition: background 0.3s;
                }
                .button:hover {
                    background: #3b82f6;
                }
                .button.stop {
                    background: #dc2626;
                }
                .button.stop:hover {
                    background: #b91c1c;
                }
                .modal {
                    display: none;
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.5);
                    justify-content: center;
                    align-items: center;
                    z-index: 1000;
                }
                .modal-content {
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    width: 90%;
                    max-width: 500px;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
                }
                .modal-content h2 {
                    font-size: 1.5em;
                    color: #1e3a8a;
                    margin-bottom: 15px;
                }
                .modal-content form {
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                }
                .modal-content input[type="text"],
                .modal-content input[type="file"] {
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    width: 100%;
                    box-sizing: border-box;
                }
                .modal-content button {
                    padding: 10px;
                    background: #1e3a8a;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }
                .modal-content button:hover {
                    background: #3b82f6;
                }
                .close {
                    float: right;
                    font-size: 1.5em;
                    cursor: pointer;
                    color: #1e3a8a;
                }
                .message {
                    margin: 10px 0;
                    padding: 10px;
                    border-radius: 5px;
                }
                .message.success {
                    background: #d1fae5;
                    color: #065f46;
                }
                .message.error {
                    background: #fee2e2;
                    color: #991b1b;
                }
                @media (max-width: 600px) {
                    h1 { font-size: 1.5em; }
                    .button { font-size: 0.9em; padding: 10px 20px; }
                    .modal-content { width: 95%; }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Live Face Recognition</h1>
                <div id="video-container">
                    <canvas id="canvas" width="800" height="600"></canvas>
                </div>
                <button class="button" onclick="startRecognition()">Start Recognition</button>
                <button class="button stop" onclick="stopRecognition()" style="display: none;">Stop Recognition</button>
                <button class="button" onclick="openModal()">Enroll New Face</button>
                <div id="message" class="message" style="display: none;"></div>
            </div>

            <!-- Modal for Enrollment Options -->
            <div id="enrollModal" class="modal">
                <div class="modal-content">
                    <span class="close" onclick="closeModal()">×</span>
                    <h2>Enroll New Face</h2>
                    <button class="button" onclick="showCaptureForm()">Capture 50 Images from Webcam</button>
                    <button class="button" onclick="showUploadForm()">Upload Images from Folder</button>

                    <!-- Capture Form -->
                    <form id="captureForm" style="display: none;" onsubmit="captureImages(event)">
                        <input type="text" id="captureName" placeholder="Enter Name" required>
                        <button type="submit">Start Capture</button>
                    </form>

                    <!-- Upload Form -->
                    <form id="uploadForm" style="display: none;" enctype="multipart/form-data" onsubmit="uploadImages(event)">
                        <input type="text" id="uploadName" placeholder="Enter Name" required>
                        <input type="file" id="uploadFiles" multiple accept="image/*" required>
                        <button type="submit">Upload Images</button>
                    </form>
                    <div id="enrollMessage" class="message" style="display: none;"></div>
                </div>
            </div>

            <script>
                let recognitionActive = false;
                let video = null;
                let captureInterval = null;

                async function startRecognition() {
                    recognitionActive = true;
                    document.querySelector('.button').style.display = 'none';
                    document.querySelector('.button.stop').style.display = 'inline-block';
                    video = document.createElement('video');
                    video.autoplay = true;
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                        video.srcObject = stream;
                        processFrames();
                    } catch (e) {
                        showMessage('message', 'Failed to access webcam: ' + e, 'error');
                    }
                }

                function stopRecognition() {
                    recognitionActive = false;
                    document.querySelector('.button').style.display = 'inline-block';
                    document.querySelector('.button.stop').style.display = 'none';
                    if (video) {
                        video.srcObject.getTracks().forEach(track => track.stop());
                        video = null;
                    }
                }

                async function processFrames() {
                    if (!recognitionActive || !video) return;
                    const canvas = document.getElementById('canvas');
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    const frame = canvas.toDataURL('image/jpeg');
                    try {
                        const response = await fetch('/process_frame', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ frame })
                        });
                        const result = await response.json();
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        result.faces.forEach(face => {
                            ctx.strokeStyle = 'green';
                            ctx.lineWidth = 2;
                            ctx.strokeRect(face.x, face.y, face.w, face.h);
                            ctx.fillStyle = 'green';
                            ctx.font = '20px Arial';
                            ctx.fillText(`${face.name} (${face.similarity.toFixed(2)})`, face.x, face.y - 10);
                        });
                    } catch (e) {
                        console.error('Frame processing failed:', e);
                    }
                    setTimeout(processFrames, 100);
                }

                function openModal() {
                    document.getElementById('enrollModal').style.display = 'flex';
                    document.getElementById('captureForm').style.display = 'none';
                    document.getElementById('uploadForm').style.display = 'none';
                    document.getElementById('enrollMessage').style.display = 'none';
                }

                function closeModal() {
                    document.getElementById('enrollModal').style.display = 'none';
                    if (captureInterval) {
                        clearInterval(captureInterval);
                        captureInterval = null;
                    }
                }

                function showCaptureForm() {
                    document.getElementById('captureForm').style.display = 'block';
                    document.getElementById('uploadForm').style.display = 'none';
                }

                function showUploadForm() {
                    document.getElementById('uploadForm').style.display = 'block';
                    document.getElementById('captureForm').style.display = 'none';
                }

                async function captureImages(event) {
                    event.preventDefault();
                    const name = document.getElementById('captureName').value.trim();
                    if (!name) {
                        showMessage('enrollMessage', 'Please enter a name.', 'error');
                        return;
                    }
                    if (!video) {
                        showMessage('enrollMessage', 'Start recognition to enable webcam.', 'error');
                        return;
                    }

                    showMessage('enrollMessage', 'Capturing 50 images...', 'success');
                    let successCount = 0;
                    let captureCount = 0;

                    captureInterval = setInterval(async () => {
                        if (captureCount >= 50) {
                            clearInterval(captureInterval);
                            captureInterval = null;
                            try {
                                await fetch('/save_embeddings', { method: 'POST' });
                                showMessage('enrollMessage', `Enrolled ${successCount}/50 images for ${name}.`, 'success');
                                setTimeout(closeModal, 2000);
                            } catch (e) {
                                showMessage('enrollMessage', `Failed to save embeddings: ${e}`, 'error');
                            }
                            return;
                        }

                        const canvas = document.createElement('canvas');
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        canvas.getContext('2d').drawImage(video, 0, 0);
                        const frame = canvas.toDataURL('image/jpeg');

                        try {
                            const response = await fetch('/enroll_frame', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ frame, name })
                            });
                            const result = await response.json();
                            if (result.success) {
                                successCount++;
                            }
                        } catch (e) {
                            console.error('Capture failed:', e);
                        }
                        captureCount++;
                    }, 500);
                }

                async function uploadImages(event) {
                    event.preventDefault();
                    const name = document.getElementById('uploadName').value.trim();
                    const files = document.getElementById('uploadFiles').files;
                    if (!name || files.length === 0) {
                        showMessage('enrollMessage', 'Please enter a name and select files.', 'error');
                        return;
                    }

                    const formData = new FormData();
                    formData.append('name', name);
                    for (let file of files) {
                        formData.append('files', file);
                    }

                    showMessage('enrollMessage', 'Uploading images...', 'success');
                    const response = await fetch('/enroll_upload', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    showMessage('enrollMessage', result.message, response.ok ? 'success' : 'error');
                    if (response.ok) {
                        setTimeout(closeModal, 2000);
                    }
                }

                function showMessage(elementId, message, type) {
                    const messageDiv = document.getElementById(elementId);
                    messageDiv.textContent = message;
                    messageDiv.className = `message ${type}`;
                    messageDiv.style.display = 'block';
                }
            </script>
        </body>
        </html>
    ''')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    frame_data = data['frame'].split(',')[1]
    img_bytes = base64.b64decode(frame_data)
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    results = []
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_tensor = transform(face_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(face_tensor)
            embedding = F.normalize(embedding, p=2, dim=1)
        person_name, similarity = find_match(embedding, embeddings_db)
        results.append({'x': x, 'y': y, 'w': w, 'h': h, 'name': person_name, 'similarity': similarity})
    return jsonify({'faces': results})

@app.route('/enroll_frame', methods=['POST'])
def enroll_frame():
    data = request.get_json()
    frame_data = data['frame'].split(',')[1]
    person_name = data['name']
    img_bytes = base64.b64decode(frame_data)
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    success, message = process_image_for_enrollment(img, person_name)
    return jsonify({'success': success, 'message': message})





@app.route('/save_embeddings', methods=['POST'])
def save_embeddings():
    try:
        torch.save(embeddings_db, 'mobilefacenet_embeddings1.pth')
        logger.info("Embeddings saved successfully")
        return jsonify({'message': 'Embeddings saved successfully'})
    except Exception as e:
        logger.error(f"Failed to save embeddings: {str(e)}")
        return jsonify({'message': f'Failed to save embeddings: {str(e)}'}), 500

@app.route('/enroll_upload', methods=['POST'])
def enroll_upload():
    person_name = request.form.get('name', '').strip()
    if not person_name:
        logger.error("Enrollment upload failed: Name is required.")
        return jsonify({'message': 'Name is required.'}), 400

    if 'files' not in request.files:
        logger.error("Enrollment upload failed: No files uploaded.")
        return jsonify({'message': 'No files uploaded.'}), 400

    files = request.files.getlist('files')
    success_count = 0
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = cv2.imread(file_path)
        if img is None:
            logger.warning(f"Failed to read image: {filename}")
            continue

        success, message = process_image_for_enrollment(img, person_name)
        if success:
            success_count += 1

        os.remove(file_path)

    try:
        torch.save(embeddings_db, 'mobilefacenet_embeddings1.pth')
        logger.info(f"Enrolled {success_count}/{len(files)} images for {person_name}")
    except Exception as e:
        logger.error(f"Failed to save embeddings: {str(e)}")
        return jsonify({'message': f'Failed to save embeddings: {str(e)}'}), 500

    return jsonify({'message': f'Enrolled {success_count}/{len(files)} images successfully for {person_name}.'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
