"""
Emotion Detection với Camera Realtime
Sử dụng webcam để phát hiện cảm xúc trực tiếp
"""

import cv2
from ultralytics import YOLO
import argparse

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

def run_camera(model_path='weights/best2.pt'):
    """Chạy camera realtime với YOLO model"""
    
    print("="*70)
    print("📹 EMOTION DETECTION - CAMERA REALTIME")
    print("="*70)
    print(f"\n🤖 Đang load model: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    print("✓ Model loaded!\n")
    
    # Mở camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Không thể mở camera!")
        return
    
    # Cấu hình camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("📹 Camera đã sẵn sàng!")
    print("\n⌨️  Controls:")
    print("   • ESC hoặc Q: Thoát")
    print("   • S: Chụp ảnh")
    print("   • SPACE: Pause/Resume")
    print("\n" + "="*70)
    
    paused = False
    frame_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("❌ Không thể đọc frame!")
                break
            
            # Detect emotions
            results = model(frame, verbose=False)
            
            # Vẽ bounding boxes
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    emotion = model.names[cls]
                    
                    # Màu theo cảm xúc
                    color = EMOTION_COLORS.get(emotion.lower(), (255, 255, 255))
                    
                    # Vẽ bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Label với background
                    label = f"{emotion.upper()} {conf*100:.1f}%"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Thông tin
            cv2.putText(frame, f"Model: {model_path.split('/')[-1]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "ESC/Q: Exit | S: Save | SPACE: Pause", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Hiển thị
        if paused:
            cv2.putText(frame, "PAUSED", (frame.shape[1]//2 - 100, frame.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        cv2.imshow('Emotion Detection - Camera', frame)
        
        # Xử lý phím
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27 or key == ord('q') or key == ord('Q'):  # ESC hoặc Q
            print("\n👋 Đang thoát...")
            break
        elif key == ord('s') or key == ord('S'):  # Chụp ảnh
            filename = f'camera_capture_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"📸 Đã lưu: {filename}")
            frame_count += 1
        elif key == ord(' '):  # SPACE - pause/resume
            paused = not paused
            status = "PAUSED" if paused else "RESUMED"
            print(f"⏸️  {status}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Đã đóng camera")
    print("="*70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Emotion Detection Camera Realtime')
    parser.add_argument('--model', type=str, default='weights/best2.pt',
                       help='Path to YOLO model (default: weights/best2.pt)')
    
    args = parser.parse_args()
    
    print("\n🎯 Available models:")
    print("   • weights/best2.pt (recommended)")
    print("   • weights/last2.pt")
    print("   • weights/best.pt")
    print("   • weights/last.pt")
    print("   • yolov8n.pt")
    print("   • yolo11n.pt")
    
    run_camera(args.model)
