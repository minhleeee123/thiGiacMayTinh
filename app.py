"""
Web app đơn giản cho Emotion Detection
Sử dụng Flask để tạo web interface
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import base64
from pathlib import Path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Tạo thư mục nếu chưa có
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Kiểm tra file extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load models (lazy loading)
models_cache = {}

def get_model(model_name):
    """Load model với caching"""
    if model_name not in models_cache:
        if model_name == 'deepface':
            models_cache[model_name] = 'deepface'
        else:
            model_path = f'weights/{model_name}' if model_name.endswith('.pt') else model_name
            models_cache[model_name] = YOLO(model_path)
    return models_cache[model_name]

# Màu cho các cảm xúc
EMOTION_COLORS = {
    'angry': (0, 0, 255),
    'anger': (0, 0, 255),
    'disgust': (0, 255, 255),
    'fear': (255, 0, 255),
    'happy': (0, 255, 0),
    'sad': (255, 0, 0),
    'surprise': (0, 165, 255),
    'neutral': (128, 128, 128),
    'content': (147, 20, 255)
}

def process_with_yolo(image_path, model_name):
    """Xử lý ảnh với YOLO model"""
    model = get_model(model_name)
    img = cv2.imread(image_path)
    
    results = model(img)
    
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            emotion = model.names[cls]
            
            # Vẽ bounding box
            color = EMOTION_COLORS.get(emotion.lower(), (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{emotion.upper()} {conf*100:.1f}%"
            cv2.putText(img, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            detections.append({
                'emotion': emotion,
                'confidence': f"{conf*100:.1f}%",
                'bbox': [x1, y1, x2, y2]
            })
    
    # Lưu kết quả
    result_filename = f"result_{os.path.basename(image_path)}"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_path, img)
    
    return result_filename, detections

def process_with_deepface(image_path):
    """Xử lý ảnh với DeepFace"""
    img = cv2.imread(image_path)
    
    try:
        results = DeepFace.analyze(img, actions=['emotion', 'age', 'gender'], 
                                   enforce_detection=False, detector_backend='opencv')
        
        if not isinstance(results, list):
            results = [results]
        
        detections = []
        for face in results:
            region = face.get('region', {})
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            
            emotion = face['dominant_emotion']
            age = face.get('age', 'N/A')
            gender = face.get('dominant_gender', 'N/A')
            
            # Vẽ bounding box
            color = EMOTION_COLORS.get(emotion.lower(), (255, 255, 255))
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Label
            label = f"{emotion.upper()}"
            cv2.putText(img, label, (x, y - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(img, f"Age: {age}, {gender}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            detections.append({
                'emotion': emotion,
                'age': age,
                'gender': gender,
                'bbox': [x, y, x+w, y+h]
            })
        
        # Lưu kết quả
        result_filename = f"result_{os.path.basename(image_path)}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, img)
        
        return result_filename, detections
        
    except Exception as e:
        return None, [{'error': str(e)}]

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint để xử lý ảnh"""
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file'}), 400
    
    file = request.files['file']
    model_name = request.form.get('model', 'best2.pt')
    
    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File không được hỗ trợ. Chỉ chấp nhận: PNG, JPG, JPEG, GIF, BMP, WEBP'}), 400
    
    if file:
        # Lưu file upload
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Xử lý ảnh
        try:
            if model_name == 'deepface':
                result_filename, detections = process_with_deepface(filepath)
            else:
                result_filename, detections = process_with_yolo(filepath, model_name)
            
            if result_filename:
                return jsonify({
                    'success': True,
                    'result_image': f'/results/{result_filename}',
                    'detections': detections,
                    'model': model_name
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Không thể xử lý ảnh',
                    'detections': detections
                }), 500
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

@app.route('/results/<filename>')
def result_file(filename):
    """Serve result images"""
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/video_feed')
def video_feed():
    """Video streaming route - trả về video stream"""
    model_name = request.args.get('model', 'best2.pt')
    return Response(generate_frames(model_name),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames(model_name):
    """Generator function để stream video frames"""
    # Load model
    if model_name == 'deepface':
        use_deepface = True
        model = None
    else:
        use_deepface = False
        model = get_model(model_name)
    
    # Mở camera
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        # Thử camera index khác
        camera = cv2.VideoCapture(1)
        if not camera.isOpened():
            print("❌ Không thể mở camera!")
            return
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"✓ Camera đã mở - đang stream với model: {model_name}")
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("⚠️ Không đọc được frame")
                break
            
            # Process frame
            try:
                if use_deepface:
                    try:
                        results = DeepFace.analyze(frame, actions=['emotion'], 
                                                 enforce_detection=False, detector_backend='opencv')
                        if not isinstance(results, list):
                            results = [results]
                        
                        for face in results:
                            region = face.get('region', {})
                            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
                            emotion = face['dominant_emotion']
                            
                            color = EMOTION_COLORS.get(emotion.lower(), (255, 255, 255))
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(frame, emotion.upper(), (x, y - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    except Exception as e:
                        pass
                else:
                    # YOLO detection
                    results = model(frame, verbose=False)
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            emotion = model.names[cls]
                            
                            color = EMOTION_COLORS.get(emotion.lower(), (255, 255, 255))
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            label = f"{emotion.upper()} {conf*100:.0f}%"
                            cv2.putText(frame, label, (x1, y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except Exception as e:
                print(f"⚠️ Lỗi xử lý frame: {e}")
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        camera.release()

if __name__ == '__main__':
    print("="*70)
    print("🚀 EMOTION DETECTION WEB APP")
    print("="*70)
    print("\n📌 Mở trình duyệt tại: http://localhost:5000")
    print("\n💡 Models có sẵn:")
    print("   • YOLOv8n (base model)")
    print("   • YOLO11n (base model)")
    print("   • best.pt (custom emotion)")
    print("   • best2.pt (custom emotion)")
    print("   • last.pt (custom emotion)")
    print("   • last2.pt (custom emotion)")
    print("   • DeepFace (emotion + age + gender)")
    print("\n" + "="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
