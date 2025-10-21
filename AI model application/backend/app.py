"""
YOLOv8 Object Detection Flask API
Real-time object detection uchun REST API server
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from PIL import Image
import io
import socket

app = Flask(__name__)
CORS(app)  # Telefon browseridan ulanish uchun

# YOLOv8 modelini yuklash
print("🔄 YOLO modelini yuklash boshlandi...")
model = YOLO('yolov8n.pt')  # 'n' - nano (eng yengil), 's', 'm', 'l', 'x' ham bor
print("✅ Model muvaffaqiyatli yuklandi!")


def get_local_ip():
    """
    Laptopning local IP manzilini olish
    Bu IP orqali telefon ulanadi
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


@app.route('/')
def home():
    """API ishlayotganini tekshirish"""
    return jsonify({
        "status": "active",
        "message": "YOLOv8 Detection API ishlayapti!",
        "server_ip": get_local_ip(),
        "endpoints": {
            "/detect": "POST - Rasm yuborish va detection olish",
            "/health": "GET - Server holatini tekshirish"
        }
    })


@app.route('/health')
def health():
    """Server sog'ligini tekshirish"""
    return jsonify({
        "status": "healthy",
        "models": "YOLOv8n",
        "ready": True
    })


@app.route('/detect', methods=['POST'])
def detect_objects():
    """
    Real-time object detection
    Telefon kamerasidan base64 rasmni qabul qilib, detection natijasini qaytaradi
    """
    try:
        # Base64 rasmni qabul qilish
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "Rasm topilmadi"}), 400
        
        # Base64 ni decode qilish
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        
        # PIL Image ga o'girish
        image = Image.open(io.BytesIO(image_bytes))
        
        # RGB formatga o'tkazish (agar kerak bo'lsa)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # NumPy array ga o'girish (OpenCV uchun)
        img_array = np.array(image)
        
        # YOLOv8 bilan detection
        results = model(img_array, conf=0.25, verbose=False)  # confidence threshold 0.25
        
        # Natijalarni tahlil qilish
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Koordinatalar
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Confidence va class
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 2),
                    "bbox": {
                        "x1": round(x1),
                        "y1": round(y1),
                        "x2": round(x2),
                        "y2": round(y2)
                    }
                })
        
        # Natijani qaytarish
        return jsonify({
            "success": True,
            "detections": detections,
            "count": len(detections),
            "image_size": {
                "width": image.width,
                "height": image.height
            }
        })
    
    except Exception as e:
        print(f"❌ Xatolik: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/classes')
def get_classes():
    """YOLOv8 aniqlashi mumkin bo'lgan obyektlar ro'yxati"""
    return jsonify({
        "classes": model.names,
        "total": len(model.names)
    })


if __name__ == '__main__':
    local_ip = get_local_ip()
    port = 5000
    
    print("\n" + "="*60)
    print("🚀 YOLOv8 Detection Server ishga tushdi!")
    print("="*60)
    print(f"📱 Telefonda ochish uchun: http://{local_ip}:{port}")
    print(f"💻 Laptopda ochish uchun: http://localhost:{port}")
    print("="*60)
    print("⚠️  DIQQAT: Telefon va laptop bir xil Wi-Fi da bo'lishi kerak!")
    print("="*60 + "\n")
    
    # Server ishga tushirish
    app.run(
        host='0.0.0.0',  # Barcha network interfacelardan ulanishga ruxsat
        port=port,
        debug=False,  # Production uchun False qiling
        threaded=True  # Bir nechta request birga ishlatish uchun
    )
