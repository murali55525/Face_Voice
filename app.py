import face_recognition
import numpy as np
import psycopg2
from psycopg2.extras import DictCursor
import os
from cryptography.fernet import Fernet
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import pickle
import logging
from pydub import AudioSegment
import tempfile
import librosa
import mediapipe as mp
from datetime import datetime
import json
import warnings

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:8000", "https://face-voice.vercel.app", "https://new-frontend-url.com"],
        "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Range", "X-Content-Range"],
        "supports_credentials": True,
        "max_age": 86400
    }
})
# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Create data directories
DATA_DIR = os.path.join(os.path.dirname(__file__), 'biometric_data')
USERS_DIR = os.path.join(DATA_DIR, 'users')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(USERS_DIR, exist_ok=True)

# Initialize MediaPipe Face Mesh and Face Detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.4,  # Lowered for better detection
    min_tracking_confidence=0.4,
    refine_landmarks=True
)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.4)  # Lowered threshold

# Load Haar Cascade for face detection
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# PostgreSQL Database Configuration
DB_CONFIG = {
    'dbname': 'biometric_db',
    'user': 'postgres',
    'password': os.getenv('PG_PASSWORD', 'Biometric2025!'),
    'host': 'localhost',
    'port': '5432'
}

def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.debug("PostgreSQL database connection established successfully")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL database: {str(e)}")
        raise

def migrate_database(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    face_model BYTEA,
                    voice_model BYTEA,
                    face_encoding BYTEA,
                    iris_encoding BYTEA,
                    voice_encoding BYTEA,
                    face_image BYTEA,
                    voice_audio BYTEA,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'users'
            """)
            columns = [col[0] for col in cur.fetchall()]
            for col in ['face_model', 'voice_model', 'face_encoding', 'iris_encoding', 
                       'voice_encoding', 'face_image', 'voice_audio']:
                if col not in columns:
                    cur.execute(f"ALTER TABLE users ADD COLUMN {col} BYTEA")
                    logger.debug(f"Added {col} column")
            conn.commit()
            logger.debug("Database schema migration completed")
    except Exception as e:
        logger.error(f"Failed to migrate database schema: {str(e)}")
        conn.rollback()
        raise

try:
    conn = get_db_connection()
    migrate_database(conn)
    logger.debug("Database connection initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize database: {str(e)}")
    raise

# Load deepfake detection models
try:
    face_deepfake_model = ViTForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")
    face_processor = ViTImageProcessor.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")
    audio_deepfake_model = Wav2Vec2ForSequenceClassification.from_pretrained("mo-thecreator/Deepfake-audio-detection")
    audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("mo-thecreator/Deepfake-audio-detection")
    logger.debug("Deepfake detection models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load deepfake detection models: {str(e)}")
    raise

class BiometricClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim=64):
        super(BiometricClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss_same = label * torch.pow(euclidean_distance, 2)
        loss_diff = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return torch.mean(loss_same + loss_diff)

def save_model(model):
    try:
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return None

def load_model(state_dict_bytes, input_dim):
    try:
        model = BiometricClassifier(input_dim)
        buffer = io.BytesIO(state_dict_bytes)
        model.load_state_dict(torch.load(buffer))
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image_for_low_light(image_np):
    """Enhance image for detection, preserving details"""
    try:
        # Convert to grayscale for brightness analysis
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)
        
        # Apply histogram equalization if brightness is low
        if mean_brightness < 100:
            lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.equalizeHist(l)
            lab = cv2.merge((l, a, b))
            image_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Apply adaptive sharpening to enhance edges
        blurred = cv2.GaussianBlur(image_np, (0, 0), 3)
        image_np = cv2.addWeighted(image_np, 1.5, blurred, -0.5, 0)
        
        # Apply slight gamma correction
        gamma = 1.2 if mean_brightness < 80 else 1.0
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        image_np = cv2.LUT(image_np, table)
        
        # Denoise to reduce noise
        image_np = cv2.fastNlMeansDenoisingColored(image_np, None, 10, 10, 7, 21)
        
        logger.debug(f"Image preprocessed, mean brightness: {mean_brightness:.2f}")
        return image_np, mean_brightness
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        return image_np, None

def process_base64_image(base64_string):
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)
        
        # Resize image if too large
        max_dimension = 1280
        height, width = image_np.shape[:2]
        if height > max_dimension or width > max_dimension:
            scale = max_dimension / max(height, width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            image_np = cv2.resize(image_np, (new_width, new_height))
            logger.debug(f"Image resized from {width}x{height} to {new_width}x{new_height}")
        
        # Enhance image quality
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_np = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Calculate brightness
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        
        logger.debug(f"Image processed - Shape: {image_np.shape}, Brightness: {brightness:.2f}")
        return image_np, brightness
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None, None

def validate_audio_file(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        if audio.duration_seconds < 1.0:
            logger.error("Audio file too short")
            return False
        if audio.frame_rate not in [16000, 44100, 48000]:
            logger.error(f"Unsupported sample rate: {audio.frame_rate}")
            return False
        logger.debug("Audio file validated successfully")
        return True
    except Exception as e:
        logger.error(f"Invalid audio file: {str(e)}")
        return False

def process_audio_file(audio_data):
    try:
        temp_webm = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
        with open(temp_webm, "wb") as f:
            f.write(audio_data)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        audio = AudioSegment.from_file(temp_webm, format="webm")
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(temp_wav, format="wav")
        if not validate_audio_file(temp_wav):
            os.remove(temp_webm)
            os.remove(temp_wav)
            return None
        os.remove(temp_webm)
        return temp_wav
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return None

def detect_face(image_np):
    """Detect face using multiple methods with higher sensitivity"""
    try:
        # Method 1: face_recognition with extreme sensitivity
        face_locations = face_recognition.face_locations(
            image_np,
            model="hog",  # Using HOG model for better performance
            number_of_times_to_upsample=2  # Increased sensitivity
        )
        if face_locations:
            logger.debug("Face detected with face_recognition")
            return face_locations

        # Method 2: MediaPipe Face Detection with lower confidence threshold
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(
            min_detection_confidence=0.3  # Lowered threshold for better detection
        ) as face_detection:
            results = face_detection.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            if results.detections:
                face_locations = []
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w = image_np.shape[:2]
                    x_min = max(0, int(bbox.xmin * w))
                    y_min = max(0, int(bbox.ymin * h))
                    width = min(int(bbox.width * w), w - x_min)
                    height = min(int(bbox.height * h), h - y_min)
                    face_locations.append((y_min, x_min + width, y_min + height, x_min))
                logger.debug("Face detected with MediaPipe")
                return face_locations

        # Method 3: Haar Cascade with adjusted parameters
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # More granular scaling
            minNeighbors=3,    # Reduced threshold
            minSize=(20, 20)   # Smaller minimum face size
        )
        if len(faces) > 0:
            face_locations = []
            for (x, y, w, h) in faces:
                face_locations.append((y, x + w, y + h, x))
            logger.debug("Face detected with Haar Cascade")
            return face_locations

        # If no face detected, try one last time with even lower thresholds
        faces = haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.01,  # Extremely fine-grained scaling
            minNeighbors=2,    # Minimum threshold
            minSize=(10, 10)   # Very small minimum face size
        )
        if len(faces) > 0:
            face_locations = []
            for (x, y, w, h) in faces:
                face_locations.append((y, x + w, y + h, x))
            logger.debug("Face detected with final attempt")
            return face_locations

        # Enhanced error message with possible solutions
        logger.warning("No face detected - Image shape: {}, Brightness: {}".format(
            image_np.shape, np.mean(gray)))
        return []
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        return []

def extract_iris_features(image):
    try:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not results.multi_face_landmarks:
            return None, "No face landmarks detected"
        landmarks = results.multi_face_landmarks[0].landmark
        LEFT_IRIS = [474, 475, 476, 477]
        RIGHT_IRIS = [469, 470, 471, 472]
        iris_features = []
        for indices in [LEFT_IRIS, RIGHT_IRIS]:
            iris_points = []
            for idx in indices:
                if idx >= len(landmarks):
                    return None, "Invalid iris landmark index"
                lm = landmarks[idx]
                iris_points.append([lm.x * image.shape[1], lm.y * image.shape[0]])
            iris_points = np.array(iris_points, dtype=np.float32)
            center = np.mean(iris_points, axis=0)
            distances = np.linalg.norm(iris_points - center, axis=1)
            normalized_distances = distances / np.max(distances)
            iris_features.extend(normalized_distances)
        iris_features = np.array(iris_features, dtype=np.float32)
        if np.any(np.isnan(iris_features)) or np.any(np.isinf(iris_features)):
            return None, "Invalid iris features"
        norm = np.linalg.norm(iris_features)
        if norm == 0:
            return None, "Zero norm iris features"
        iris_features = iris_features / norm
        return iris_features, "Iris features extracted"
    except Exception as e:
        logger.error(f"Iris feature extraction error: {str(e)}")
        return None, f"Iris feature extraction failed: {str(e)}"

def check_face_liveness(image):
    try:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not results.multi_face_landmarks:
            return False, "No face landmarks detected"
        landmarks = results.multi_face_landmarks[0].landmark
        FACIAL_LANDMARKS = {
            'nose_tip': 1,
            'left_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'right_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'left_ear': 234,
            'right_ear': 454,
            'mouth': [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37],
            'left_iris': 474,
            'right_iris': 469
        }
        for feature, indices in FACIAL_LANDMARKS.items():
            if isinstance(indices, list):
                if not all(0 <= idx < len(landmarks) for idx in indices):
                    return False, f"Invalid landmark index for {feature}"
            else:
                if not 0 <= indices < len(landmarks):
                    return False, f"Invalid landmark index for {feature}"
        nose_depth = landmarks[FACIAL_LANDMARKS['nose_tip']].z
        left_ear_pos = landmarks[FACIAL_LANDMARKS['left_ear']]
        right_ear_pos = landmarks[FACIAL_LANDMARKS['right_ear']]
        ear_distance = abs(left_ear_pos.x - right_ear_pos.x)
        if ear_distance == 0:
            return False, "Invalid ear distance detected"
        depth_ratio = abs(nose_depth) / ear_distance
        def get_eye_height(eye_points):
            top_point = min(landmarks[i].y for i in eye_points)
            bottom_point = max(landmarks[i].y for i in eye_points)
            return abs(top_point - bottom_point)
        left_eye_height = get_eye_height(FACIAL_LANDMARKS['left_eye'])
        right_eye_height = get_eye_height(FACIAL_LANDMARKS['right_eye'])
        left_iris_z = landmarks[FACIAL_LANDMARKS['left_iris']].z
        right_iris_z = landmarks[FACIAL_LANDMARKS['right_iris']].z
        iris_z_diff = abs(left_iris_z - right_iris_z)
        face_rotation = abs(left_ear_pos.z - right_ear_pos.z)
        checks = {
            "depth": 0.1 < depth_ratio < 1.0,
            "eyes_open": min(left_eye_height, right_eye_height) > 0.01,
            "iris_alignment": iris_z_diff < 0.05,
            "face_angle": face_rotation < 0.5
        }
        failed_checks = [k for k, v in checks.items() if not v]
        logger.debug(f"Liveness measurements - depth_ratio: {depth_ratio:.3f}, "
                    f"eye_heights: {left_eye_height:.3f}/{right_eye_height:.3f}, "
                    f"iris_z_diff: {iris_z_diff:.3f}, face_rotation: {face_rotation:.3f}")
        if failed_checks:
            return False, f"Liveness check failed: {', '.join(failed_checks)}"
        return True, "Liveness check passed"
    except Exception as e:
        logger.error(f"Liveness check error: {str(e)}")
        return False, f"Liveness check failed: {str(e)}"

def save_user_data(user_id, face_image_np, voice_wav=None):
    try:
        user_dir = os.path.join(USERS_DIR, str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        face_path = os.path.join(user_dir, 'face.jpg')
        cv2.imwrite(face_path, face_image_np)
        if voice_wav:
            voice_path = os.path.join(user_dir, 'voice.wav')
            AudioSegment.from_wav(voice_wav).export(voice_path, format='wav')
        metadata = {
            'enrolled_at': datetime.now().isoformat(),
            'face_shape': face_image_np.shape,
            'last_verified': None
        }
        with open(os.path.join(user_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        return True
    except Exception as e:
        logger.error(f"Error saving user data: {str(e)}")
        return False

def train_biometric_model(features, input_dim, user_id, epochs=30):
    try:
        model = BiometricClassifier(input_dim)
        criterion = ContrastiveLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        features_tensor = features_tensor / torch.norm(features_tensor)
        positive_samples = [features_tensor]
        for _ in range(5):
            noise = torch.randn_like(features_tensor) * 0.02
            augmented = features_tensor + noise
            augmented = augmented / torch.norm(augmented)
            positive_samples.append(augmented)
        positive_samples = torch.cat(positive_samples, dim=0)
        negative_samples = torch.randn(5, input_dim)
        negative_samples = negative_samples / torch.norm(negative_samples, dim=1, keepdim=True)
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(positive_samples)):
                for j in range(i + 1, len(positive_samples)):
                    optimizer.zero_grad()
                    out1 = model(positive_samples[i:i+1])
                    out2 = model(positive_samples[j:j+1])
                    loss = criterion(out1, out2, torch.tensor(1.0))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            for pos in positive_samples:
                for neg in negative_samples:
                    optimizer.zero_grad()
                    out1 = model(pos.unsqueeze(0))
                    out2 = model(neg.unsqueeze(0))
                    loss = criterion(out1, out2, torch.tensor(0.0))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error in train_biometric_model: {str(e)}")
        return None

def enroll_user(user_id, face_image_np, voice_file=None):
    global conn
    voice_wav = None
    temp_voice_file = None
    try:
        if face_image_np.dtype != np.uint8:
            face_image_np = face_image_np.astype(np.uint8)
        face_image_np = cv2.resize(face_image_np, (128, 128))
        face_locations = detect_face(face_image_np)
        if not face_locations:
            logger.error("No face detected")
            return False, "No face detected. Please ensure your face is centered and well-lit."
        face_encoding = face_recognition.face_encodings(face_image_np, face_locations)[0]
        if face_encoding.shape != (128,) or np.any(np.isnan(face_encoding)) or np.any(np.isinf(face_encoding)):
            logger.error("Invalid face encoding")
            return False, "Invalid face encoding"
        norm = np.linalg.norm(face_encoding)
        if norm == 0:
            logger.error("Face encoding has zero norm")
            return False, "Invalid face encoding"
        face_encoding = face_encoding / norm
        iris_encoding, iris_msg = extract_iris_features(face_image_np)
        if iris_encoding is None:
            logger.error(f"Iris encoding failed: {iris_msg}")
            return False, f"Iris encoding failed: {iris_msg}"
        is_live, liveness_msg = check_face_liveness(face_image_np)
        if not is_live:
            logger.error(f"Face liveness check failed: {liveness_msg}")
            return False, f"Face liveness check failed: {liveness_msg}"
        inputs = face_processor(images=face_image_np, return_tensors="pt")
        with torch.no_grad():
            outputs = face_deepfake_model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        real_prob = probs[:, 0].item()
        if real_prob < 0.3:
            logger.error(f"Face flagged as deepfake (real prob: {real_prob:.4f})")
            return False, "Face is deepfake"
        voice_encoding = None
        voice_model_bytes = None
        if voice_file:
            try:
                temp_voice_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
                voice_segment = AudioSegment.from_file(voice_file)
                voice_segment.export(temp_voice_file, format='wav')
                audio_data, sr = librosa.load(temp_voice_file, sr=16000, mono=True)
                audio_features = audio_feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
                with torch.no_grad():
                    audio_outputs = audio_deepfake_model(**audio_features)
                audio_probs = torch.nn.functional.softmax(audio_outputs.logits, dim=-1)
                if audio_probs[:, 0].item() < 0.5:
                    logger.error("Voice is deepfake")
                    return False, "Voice is deepfake"
                voice_encoding = np.random.randn(192)
                voice_encoding = voice_encoding / np.linalg.norm(voice_encoding)
            except Exception as e:
                logger.warning(f"Voice processing failed: {str(e)}, continuing without voice")
        face_model = train_biometric_model(face_encoding, input_dim=128, user_id=user_id)
        if face_model is None:
            logger.error("Failed to train face model")
            return False, "Failed to train face model"
        face_model_bytes = save_model(face_model)
        if face_model_bytes is None:
            logger.error("Failed to save face model")
            return False, "Failed to save face model"
        if voice_encoding is not None:
            voice_model = train_biometric_model(voice_encoding, input_dim=voice_encoding.shape[0], user_id=user_id)
            if voice_model:
                voice_model_bytes = save_model(voice_model)
                if voice_model_bytes is None:
                    logger.warning("Failed to save voice model, continuing without voice")
        if not save_user_data(user_id, face_image_np, temp_voice_file):
            return False, "Failed to save user data"
        _, face_image_bytes = cv2.imencode('.jpg', face_image_np)
        face_image_bytes = face_image_bytes.tobytes()
        voice_audio_bytes = None
        if temp_voice_file:
            with open(temp_voice_file, 'rb') as f:
                voice_audio_bytes = f.read()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users 
                (user_id, face_model, voice_model, face_encoding, iris_encoding, voice_encoding, face_image, voice_audio) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (user_id, 
                 psycopg2.Binary(face_model_bytes), 
                 psycopg2.Binary(voice_model_bytes) if voice_model_bytes else None,
                 psycopg2.Binary(pickle.dumps(face_encoding)),
                 psycopg2.Binary(pickle.dumps(iris_encoding)),
                 psycopg2.Binary(pickle.dumps(voice_encoding)) if voice_encoding is not None else None,
                 psycopg2.Binary(face_image_bytes),
                 psycopg2.Binary(voice_audio_bytes) if voice_audio_bytes else None))
            conn.commit()
            return True, "User enrolled successfully"
    except Exception as e:
        logger.error(f"Error in enroll_user: {str(e)}")
        conn.rollback()
        return False, f"Enrollment failed: {str(e)}"
    finally:
        for temp_file in [voice_file, temp_voice_file, voice_wav]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")

def authenticate_user(user_id, face_image_np, voice_file=None):
    temp_voice_file = None
    try:
        logger.debug(f"Authenticating user {user_id}, image shape: {face_image_np.shape}")
        
        # Preprocess image
        face_image_np, brightness = preprocess_image_for_low_light(face_image_np)
        face_locations = detect_face(face_image_np)
        if not face_locations:
            logger.error("No face detected")
            return False, "No face detected. Please ensure your face is centered and well-lit."
        
        if len(face_locations) > 1:
            logger.error("Multiple faces detected")
            return False, "Multiple faces detected. Please ensure only one person is in frame."
        
        # Get multiple encodings for robustness
        face_encodings = []
        for upsample in [1, 2]:
            enc = face_recognition.face_encodings(face_image_np, face_locations, num_jitters=10, model="large")[0]
            enc = enc / np.linalg.norm(enc)
            face_encodings.append(enc)
            logger.debug(f"Face encoding norm: {np.linalg.norm(enc)}")
        
        iris_encoding, iris_msg = extract_iris_features(face_image_np)
        if iris_encoding is None:
            logger.error(f"Iris encoding failed: {iris_msg}")
            return False, f"Iris authentication failed: {iris_msg}"
        
        is_live, liveness_msg = check_face_liveness(face_image_np)
        if not is_live:
            logger.error(f"Liveness check failed: {liveness_msg}")
            return False, f"Face liveness check failed: {liveness_msg}"
        
        with conn.cursor() as c:
            c.execute("SELECT face_model, voice_model, face_encoding, iris_encoding, voice_encoding FROM users WHERE user_id = %s", (user_id,))
            result = c.fetchone()
            if not result:
                logger.error(f"User {user_id} not found")
                return False, "User not found"
            
            face_model_bytes, voice_model_bytes, stored_face_enc, stored_iris_enc, stored_voice_enc = result
            stored_face_encoding = pickle.loads(stored_face_enc)
            stored_iris_encoding = pickle.loads(stored_iris_enc)
            stored_voice_encoding = pickle.loads(stored_voice_enc) if stored_voice_enc else None
        
        # Face matching
        face_distances = [face_recognition.face_distance([stored_face_encoding], enc)[0] for enc in face_encodings]
        min_face_distance = min(face_distances)
        logger.debug(f"Min face distance: {min_face_distance}")
        if min_face_distance > 0.45:
            logger.error(f"Face distance too high: {min_face_distance}")
            return False, "Face does not match stored biometric data"
        
        # Iris matching
        iris_distance = np.linalg.norm(iris_encoding - stored_iris_encoding)
        logger.debug(f"Iris distance: {iris_distance}")
        if iris_distance > 0.3:
            logger.error(f"Iris distance too high: {iris_distance}")
            return False, "Iris does not match stored biometric data"
        
        # Geometric check
        face_bbox = face_locations[0]
        face_height = face_bbox[2] - face_bbox[0]
        face_width = face_bbox[1] - face_bbox[3]
        aspect_ratio = face_height / face_width
        logger.debug(f"Face aspect ratio: {aspect_ratio}")
        if not (0.8 <= aspect_ratio <= 1.2):
            logger.error(f"Invalid face aspect ratio: {aspect_ratio}")
            return False, "Face position or angle is not optimal"
        
        # Model-based verification
        face_model = load_model(face_model_bytes, input_dim=128)
        if not face_model:
            logger.error("Failed to load face model")
            return False, "Failed to load face model"
        with torch.no_grad():
            face_input = torch.tensor(face_encodings[0], dtype=torch.float32).unsqueeze(0)
            stored_face = torch.tensor(stored_face_encoding, dtype=torch.float32).unsqueeze(0)
            face_emb = face_model(face_input)
            stored_face_emb = face_model(stored_face)
            face_distance = torch.nn.functional.pairwise_distance(face_emb, stored_face_emb).item()
            logger.debug(f"Model face distance: {face_distance}")
            if face_distance > 0.3:
                logger.error(f"Model verification failed: {face_distance}")
                return False, "Face biometric verification failed"
        
        # Voice verification (optional)
        voice_match = True
        if voice_file and stored_voice_encoding is not None:
            try:
                temp_voice_file = process_audio_file(voice_file)
                if temp_voice_file:
                    voice_encoding = np.random.randn(192)  # Replace with actual model
                    voice_encoding = voice_encoding / np.linalg.norm(voice_encoding)
                    voice_distance = np.linalg.norm(voice_encoding - stored_voice_encoding)
                    logger.debug(f"Voice distance: {voice_distance}")
                    voice_match = voice_distance < 0.7
                    if not voice_match:
                        logger.warning(f"Voice verification failed: {voice_distance}")
                else:
                    logger.warning("Voice processing failed, skipping")
            except Exception as e:
                logger.warning(f"Voice authentication failed: {str(e)}, proceeding with face-only")
        
        if face_distance <= 0.3 and voice_match:
            logger.info(f"Authentication successful for user {user_id}")
            return True, "Authentication successful"
        logger.error("Authentication failed: combined verification checks")
        return False, "Authentication failed"
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        return False, f"Authentication failed: {str(e)}"
    finally:
        if temp_voice_file and os.path.exists(temp_voice_file):
            try:
                os.remove(temp_voice_file)
            except Exception as e:
                logger.warning(f"Failed to remove temp file: {str(e)}")

@app.route('/signup', methods=['POST', 'OPTIONS'])
def signup():
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', 'https://face-voice.vercel.app')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    global conn
    try:
        data = request.json
        user_id = data.get('user_id')
        face_image_base64 = data.get('face_image')
        voice_data_base64 = data.get('voice_data')
        if not user_id or not face_image_base64:
            logger.error("Missing required fields")
            return jsonify({"success": False, "message": "Missing user_id or face_image"}), 400
        face_image_np, brightness = process_base64_image(face_image_base64)
        if face_image_np is None:
            logger.error("Invalid face image")
            message = "Invalid face image"
            if brightness is not None and brightness < 50:
                message += ". Image too dark, please improve lighting."
            return jsonify({"success": False, "message": message}), 400
        voice_file = None
        if voice_data_base64:
            if "," in voice_data_base64:
                voice_data_base64 = voice_data_base64.split(",")[1]
            voice_data = base64.b64decode(voice_data_base64)
            voice_file = process_audio_file(voice_data)
        success, message = enroll_user(user_id, face_image_np, voice_file)
        if brightness is not None and brightness < 50 and success:
            message += ". Warning: Low lighting detected, consider better lighting for optimal performance."
        return jsonify({"success": success, "message": message})
    except Exception as e:
        logger.error(f"Error in signup endpoint: {str(e)}")
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

@app.route('/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', 'https://face-voice.vercel.app')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    global conn
    try:
        data = request.json
        user_id = data.get('user_id')
        face_image_base64 = data.get('face_image')
        voice_data_base64 = data.get('voice_data')
        if not user_id or not face_image_base64:
            logger.error("Missing required fields")
            return jsonify({"success": False, "message": "Missing user_id or face_image"}), 400
        face_image_np, brightness = process_base64_image(face_image_base64)
        if face_image_np is None:
            logger.error("Invalid face image")
            message = "Invalid face image"
            if brightness is not None and brightness < 50:
                message += ". Image too dark, please improve lighting."
            return jsonify({"success": False, "message": message}), 400
        voice_file = None
        if voice_data_base64:
            if "," in voice_data_base64:
                voice_data_base64 = voice_data_base64.split(",")[1]
            voice_data = base64.b64decode(voice_data_base64)
            voice_file = process_audio_file(voice_data)
        success, message = authenticate_user(user_id, face_image_np, voice_file)
        if brightness is not None and brightness < 50 and success:
            message += ". Warning: Low lighting detected, consider better lighting for optimal performance."
        return jsonify({"success": success, "message": message})
    except Exception as e:
        logger.error(f"Error in login endpoint: {str(e)}")
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        if 'conn' in globals():
            conn.close()
            logger.debug("Database connection closed")
