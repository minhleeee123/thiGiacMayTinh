"""
Script test 3 mô hình mới: best2.pt, last2.pt, model.h5
"""

from ultralytics import YOLO
import cv2
import os
import glob
import numpy as np

print("="*80)
print("TEST 3 MÔ HÌNH MỚI")
print("="*80)

# Đường dẫn các model
models = {
    'Best2 Model': 'weights/best2.pt',
    'Last2 Model': 'weights/last2.pt',
    'Model H5': 'weights/model.h5'
}

# Thư mục ảnh
photo_dir = 'photo'
image_files = glob.glob(os.path.join(photo_dir, '*.jpg'))
print(f"\nTìm thấy {len(image_files)} ảnh trong thư mục {photo_dir}\n")

# Test từng model
for model_name, model_path in models.items():
    output_dir = f'runs/detect/{model_name.lower().replace(" ", "_")}_results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print(f"ĐANG TEST: {model_name}")
    print(f"Model path: {model_path}")
    print("="*80)
    
    try:
        # Kiểm tra file tồn tại
        if not os.path.exists(model_path):
            print(f"❌ File không tồn tại: {model_path}\n")
            continue
        
        # Load model dựa trên extension
        if model_path.endswith('.pt'):
            # YOLO model
            print(f"Đang load YOLO model...")
            model = YOLO(model_path)
            print(f"✓ Đã load model!\n")
            
            # Test trên từng ảnh
            success_count = 0
            for i, img_path in enumerate(image_files, 1):
                img_name = os.path.basename(img_path)
                print(f"[{i}/{len(image_files)}] {img_name}", end=" ")
                
                try:
                    results = model.predict(
                        source=img_path,
                        conf=0.25,
                        save=True,
                        project='runs/detect',
                        name=output_dir.split('/')[-1],
                        exist_ok=True,
                        verbose=False
                    )
                    
                    # Đếm detections
                    detections = len(results[0].boxes)
                    if detections > 0:
                        # Lấy class names
                        classes = [model.names[int(c)] for c in results[0].boxes.cls]
                        print(f"✓ ({detections} detections: {', '.join(set(classes))})")
                        success_count += 1
                    else:
                        print(f"⚠️ (không phát hiện)")
                        
                except Exception as e:
                    print(f"✗ Lỗi: {e}")
            
            print(f"\n📊 Kết quả: {success_count}/{len(image_files)} ảnh phát hiện thành công")
            print(f"📁 Lưu tại: {output_dir}\n")
            
        elif model_path.endswith('.h5'):
            # Keras/TensorFlow model
            print(f"Đang load Keras/TensorFlow model...")
            
            try:
                from tensorflow import keras
                import tensorflow as tf
                
                model = keras.models.load_model(model_path)
                print(f"✓ Đã load model H5!")
                print(f"Model input shape: {model.input_shape}")
                print(f"Model output shape: {model.output_shape}\n")
                
                # Emotion labels (giả định theo FER2013)
                emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                
                # Test trên từng ảnh
                success_count = 0
                for i, img_path in enumerate(image_files, 1):
                    img_name = os.path.basename(img_path)
                    print(f"[{i}/{len(image_files)}] {img_name}", end=" ")
                    
                    try:
                        # Đọc và preprocess ảnh
                        img = cv2.imread(img_path)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # Resize theo input model (thường 48x48 cho FER)
                        target_size = model.input_shape[1:3]
                        resized = cv2.resize(gray, target_size)
                        
                        # Normalize
                        normalized = resized / 255.0
                        reshaped = normalized.reshape(1, target_size[0], target_size[1], 1)
                        
                        # Predict
                        predictions = model.predict(reshaped, verbose=0)
                        emotion_idx = np.argmax(predictions[0])
                        confidence = predictions[0][emotion_idx]
                        
                        if emotion_idx < len(emotion_labels):
                            emotion = emotion_labels[emotion_idx]
                            print(f"✓ {emotion.upper()} ({confidence:.1%})")
                            success_count += 1
                        else:
                            print(f"⚠️ (index {emotion_idx} out of range)")
                            
                    except Exception as e:
                        print(f"✗ Lỗi: {e}")
                
                print(f"\n📊 Kết quả: {success_count}/{len(image_files)} ảnh phân loại thành công")
                print(f"⚠️ Lưu ý: Model H5 chỉ classification, không có bounding box\n")
                
            except ImportError:
                print(f"❌ Cần cài đặt TensorFlow: pip install tensorflow")
            except Exception as e:
                print(f"❌ Lỗi khi load model: {e}\n")
                
    except Exception as e:
        print(f"❌ Lỗi: {e}\n")

print("="*80)
print("ĐÃ HOÀN THÀNH TEST TẤT CẢ CÁC MÔ HÌNH")
print("="*80)
