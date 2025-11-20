from ultralytics import YOLO
import os

# Đường dẫn đến các mô hình
models = {
    # Mô hình pre-trained YOLO (phát hiện đối tượng tổng quát)
    'YOLOv8n': 'yolov8n.pt',
    'YOLOv8s': 'yolov8s.pt',
    'YOLOv8m': 'yolov8m.pt',
    'YOLO11n': 'yolo11n.pt',
    'YOLO11s': 'yolo11s.pt',
    
    # Mô hình custom của bạn (nhận diện cảm xúc)
    'Best Model (Custom)': 'weights/best.pt',
    'Last Model (Custom)': 'weights/last.pt'
}

# Mô hình phát hiện cảm xúc có sẵn trên mạng (Roboflow/Hugging Face)
# Bạn có thể thêm URL hoặc path của các mô hình này
pretrained_emotion_models = {
    # Ví dụ: Có thể tải từ Roboflow hoặc các nguồn khác
    # 'Roboflow Emotion': 'https://universe.roboflow.com/.../model.pt',
    # 'HuggingFace Emotion': 'hf://models/emotion-detection/model.pt',
}

print("="*60)
print("DANH SÁCH MÔ HÌNH PHỔ BIẾN PHÁT HIỆN CẢM XÚC TRÊN MẠNG:")
print("="*60)
print("\n1. Roboflow Universe - Human Face Emotions:")
print("   URL: https://universe.roboflow.com/emotions-dectection/human-face-emotions")
print("   → Có thể export model ở định dạng YOLOv8")
print("\n2. Kaggle - FER2013 Dataset với YOLO:")
print("   URL: https://www.kaggle.com/datasets/msambare/fer2013")
print("   → Nhiều model pre-trained có sẵn")
print("\n3. Hugging Face - Emotion Detection Models:")
print("   URL: https://huggingface.co/models?other=emotion-detection")
print("   → Có các model YOLO cho emotion detection")
print("\n4. GitHub - emotion-recognition YOLO:")
print("   URL: https://github.com/topics/emotion-recognition")
print("   → Nhiều project open-source với model đã train sẵn")
print("\n" + "="*60)
print("Để sử dụng model từ Roboflow, bạn cần:")
print("1. Truy cập link Roboflow Universe")
print("2. Export model ở định dạng YOLOv8")
print("3. Tải về và đặt vào thư mục weights/")
print("="*60 + "\n")

# Đường dẫn thư mục ảnh
photo_dir = 'photo'

# Load tất cả các mô hình
loaded_models = {}
for model_name, model_path in models.items():
    print(f"Đang load mô hình {model_name}...")
    try:
        loaded_models[model_name] = YOLO(model_path)
        print(f"  ✓ Đã load xong {model_name}!")
    except Exception as e:
        print(f"  ✗ Không thể load {model_name}: {e}")
        print(f"  → Mô hình sẽ được tự động tải về nếu chưa có...")
        continue

# Lấy danh sách tất cả ảnh trong thư mục photo
image_files = [f for f in os.listdir(photo_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"\nTìm thấy {len(image_files)} ảnh trong thư mục {photo_dir}")

# Chạy prediction với từng mô hình
for model_name, model in loaded_models.items():
    # Tạo tên thư mục kết quả từ tên model (loại bỏ khoảng trắng)
    result_folder = model_name.lower().replace(' ', '_') + '_results'
    
    print("\n" + "="*50)
    print(f"ĐANG CHẠY DỰ ĐOÁN VỚI {model_name.upper()}")
    print("="*50)
    
    for img_file in image_files:
        img_path = os.path.join(photo_dir, img_file)
        print(f"\nXử lý: {img_file}")
        try:
            results = model.predict(
                source=img_path,
                conf=0.25,
                save=True,
                project='runs/detect',
                name=result_folder,
                exist_ok=True
            )
            print(f"  ✓ Hoàn thành: {img_file}")
        except Exception as e:
            print(f"  ✗ Lỗi khi xử lý {img_file}: {e}")

print("\n" + "="*50)
print("ĐÃ CHẠY XONG TẤT CẢ!")
print("="*50)
for model_name in models.keys():
    result_folder = model_name.lower().replace(' ', '_') + '_results'
    print(f"Kết quả {model_name} được lưu tại: runs/detect/{result_folder}")
