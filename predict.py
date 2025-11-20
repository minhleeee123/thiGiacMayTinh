from ultralytics import YOLO
model_path = r'C:\Users\Admin\Downloads\TGMT cua Kien\Emotion_Detection\runs\detect\train2\weights\best.pt'
video_path = r'C:\Users\Admin\Downloads\TGMT cua Kien\Lionel Messi - WORLD CHAMPION - Movie.mp4' 
model = YOLO(model_path)
print(f"Đang chạy dự đoán trên file: {video_path}")
try:
    results = model.predict(source=video_path, 
                            show=True,    
                            conf=0.5,    
                            save=True)    
except Exception as e:
    print(f"Gặp lỗi: {e}")

print("Đã chạy xong!")