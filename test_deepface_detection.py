"""
Script sử dụng DeepFace với Face Detection + Bounding Box
Phát hiện khuôn mặt, vẽ bounding box và phân loại cảm xúc
"""

from deepface import DeepFace
import cv2
import os
import glob

print("="*70)
print("DEEPFACE - PHÁT HIỆN KHUÔN MẶT + CẢM XÚC VỚI BOUNDING BOX")
print("="*70)

# Thư mục chứa ảnh
photo_dir = 'photo'

# Lấy danh sách tất cả ảnh
image_files = glob.glob(os.path.join(photo_dir, '*.jpg'))
print(f"\nTìm thấy {len(image_files)} ảnh trong thư mục {photo_dir}")

# Tạo thư mục lưu kết quả
output_dir = 'runs/detect/deepface_with_bbox'
os.makedirs(output_dir, exist_ok=True)

print("\n" + "="*70)
print("ĐANG PHÁT HIỆN KHUÔN MẶT VÀ CẢM XÚC")
print("="*70)
print("\nDeepFace sẽ:")
print("• Phát hiện khuôn mặt và vẽ bounding box")
print("• Phân loại cảm xúc cho từng khuôn mặt")
print("• Hiển thị tuổi, giới tính")
print("• Lưu ảnh có bounding box\n")

# Màu sắc cho từng cảm xúc (BGR format)
emotion_colors = {
    'angry': (0, 0, 255),      # Đỏ
    'disgust': (0, 255, 255),  # Vàng
    'fear': (255, 0, 255),     # Tím
    'happy': (0, 255, 0),      # Xanh lá
    'sad': (255, 0, 0),        # Xanh dương
    'surprise': (0, 165, 255), # Cam
    'neutral': (128, 128, 128) # Xám
}

# Phân tích từng ảnh
for i, img_path in enumerate(image_files, 1):
    img_name = os.path.basename(img_path)
    print(f"\n[{i}/{len(image_files)}] Đang phân tích: {img_name}")
    
    try:
        # Đọc ảnh
        img = cv2.imread(img_path)
        
        # Phân tích với detector để lấy bounding box
        results = DeepFace.analyze(
            img_path=img_path,
            actions=['emotion', 'age', 'gender'],
            detector_backend='opencv',  # Có thể dùng: opencv, ssd, mtcnn, retinaface
            enforce_detection=False
        )
        
        # results có thể là list nếu có nhiều khuôn mặt
        if not isinstance(results, list):
            results = [results]
        
        print(f"  ✓ Tìm thấy {len(results)} khuôn mặt")
        
        # Vẽ bounding box và thông tin cho từng khuôn mặt
        for face_idx, result in enumerate(results, 1):
            # Lấy thông tin
            region = result['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            dominant_emotion = result['dominant_emotion']
            emotion_scores = result['emotion']
            age = result['age']
            gender = result['dominant_gender']
            confidence = emotion_scores[dominant_emotion]
            
            # Chọn màu theo cảm xúc
            color = emotion_colors.get(dominant_emotion, (255, 255, 255))
            
            # Vẽ bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Tạo label
            label = f"{dominant_emotion.upper()} {confidence:.1f}%"
            info = f"{gender}, {age}y"
            
            # Vẽ background cho text
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img, (x, y - label_size[1] - 10), (x + label_size[0] + 10, y), color, -1)
            
            # Vẽ text
            cv2.putText(img, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, info, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            print(f"    Face {face_idx}: {dominant_emotion.upper()} ({confidence:.1f}%) - {gender}, {age} tuổi")
        
        # Lưu ảnh
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, img)
        print(f"  → Đã lưu: {output_path}")
        
    except Exception as e:
        print(f"  ✗ Lỗi khi phân tích {img_name}: {e}")

print("\n" + "="*70)
print("ĐÃ HOÀN THÀNH!")
print("="*70)
print(f"Kết quả với bounding box được lưu tại: {output_dir}")
print("\nSo sánh:")
print("• YOLO: Phát hiện nhanh, real-time, có bounding box")
print("• DeepFace: Phát hiện chi tiết hơn (tuổi, giới tính), chậm hơn")
