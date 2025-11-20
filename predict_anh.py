from ultralytics import YOLO
model_path = r'C:\Users\Admin\Downloads\TGMT cua Kien\Emotion_Detection\runs\detect\train2\weights\best.pt'
model = YOLO(model_path)
test_source = r'C:\Users\Admin\Downloads\TGMT cua Kien\Emotion_Detection\R.jpg' 
print(f"Đang chạy dự đoán trên file: {test_source}")
try:
    results = model.predict(source=test_source, 
                            show=True,   
                            conf=0.2,     
                            save=True)    
except Exception as e:
    print(f"Gặp lỗi: {e}")

print("Đã chạy xong!")