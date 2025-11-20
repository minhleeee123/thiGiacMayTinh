"""
Script sử dụng DeepFace cho phát hiện cảm xúc trên khuôn mặt
DeepFace hỗ trợ nhiều model: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace
"""

from deepface import DeepFace
import os
import glob

print("="*70)
print("DEEPFACE - PHÁT HIỆN CẢM XÚC TRÊN KHUÔN MẶT")
print("="*70)

# Thư mục chứa ảnh
photo_dir = 'photo'

# Lấy danh sách tất cả ảnh
image_files = glob.glob(os.path.join(photo_dir, '*.jpg'))
print(f"\nTìm thấy {len(image_files)} ảnh trong thư mục {photo_dir}")

# Tạo thư mục lưu kết quả
output_dir = 'runs/detect/deepface_results'
os.makedirs(output_dir, exist_ok=True)

print("\n" + "="*70)
print("ĐANG PHÂN TÍCH CẢM XÚC VỚI DEEPFACE")
print("="*70)
print("\nDeepFace sẽ phát hiện:")
print("• Cảm xúc: angry, disgust, fear, happy, sad, surprise, neutral")
print("• Tuổi, giới tính, chủng tộc")
print("• Tự động download model lần đầu chạy\n")

# Phân tích từng ảnh
for i, img_path in enumerate(image_files, 1):
    img_name = os.path.basename(img_path)
    print(f"\n[{i}/{len(image_files)}] Đang phân tích: {img_name}")
    
    try:
        # Phân tích cảm xúc
        result = DeepFace.analyze(
            img_path=img_path,
            actions=['emotion', 'age', 'gender'],
            enforce_detection=False  # Không bắt buộc phải detect face
        )
        
        # Lấy kết quả (result có thể là list nếu có nhiều khuôn mặt)
        if isinstance(result, list):
            result = result[0]
        
        # Hiển thị kết quả
        dominant_emotion = result['dominant_emotion']
        emotion_scores = result['emotion']
        age = result['age']
        gender = result['dominant_gender']
        
        print(f"  ✓ Cảm xúc chủ đạo: {dominant_emotion.upper()}")
        print(f"  • Tuổi ước tính: {age}")
        print(f"  • Giới tính: {gender}")
        print(f"  • Chi tiết cảm xúc:")
        for emotion, score in emotion_scores.items():
            print(f"    - {emotion}: {score:.2f}%")
        
        # Lưu kết quả vào file text
        result_file = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"Image: {img_name}\n")
            f.write(f"Dominant Emotion: {dominant_emotion}\n")
            f.write(f"Age: {age}\n")
            f.write(f"Gender: {gender}\n")
            f.write(f"\nEmotion Scores:\n")
            for emotion, score in emotion_scores.items():
                f.write(f"  {emotion}: {score:.2f}%\n")
        
    except Exception as e:
        print(f"  ✗ Lỗi khi phân tích {img_name}: {e}")

print("\n" + "="*70)
print("ĐÃ HOÀN THÀNH!")
print("="*70)
print(f"Kết quả được lưu tại: {output_dir}")
print("\nDeepFace phát hiện 7 cảm xúc:")
print("angry, disgust, fear, happy, sad, surprise, neutral")
