# 🎯 YOLOv8 Real-Time Object Detection

Telefon kamerasi orqali real-time obyektlarni aniqlash tizimi. Laptop serverda YOLO modeli ishlab, telefonda natijalar ko'rsatiladi.

## 📋 Tizim Talablari

### Laptop (Server) uchun:
- **OS:** Windows 10/11, macOS, yoki Linux
- **Python:** 3.8 yoki yuqori
- **RAM:** Minimum 4GB (8GB tavsiya)
- **Disk:** 2GB bo'sh joy
- **Internet:** Model yuklab olish uchun (birinchi marta)

### Telefon uchun:
- **Browser:** Chrome, Safari, Firefox (zamonaviy versiya)
- **Kamera:** Old yoki orqa kamera
- **Internet:** Laptop bilan bir xil Wi-Fi tarmog'ida

## 🚀 Tezkor Ishga Tushirish

### 1-QADAM: Python Virtual Environment yaratish

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2-QADAM: Kerakli kutubxonalarni o'rnatish

```bash
pip install -r requirements.txt
```

⏳ **Diqqat:** Birinchi o'rnatish 5-10 daqiqa davom etishi mumkin (PyTorch katta hajmli).

### 3-QADAM: Serverni ishga tushirish

```bash
python app.py
```

Konsolda quyidagicha xabar ko'rinadi:
```
======================================
🚀 YOLOv8 Detection Server ishga tushdi!
======================================
📱 Telefonda ochish uchun: http://192.168.1.100:5000
💻 Laptopda ochish uchun: http://localhost:5000
======================================
```

### 4-QADAM: Telefondan ulanish

1. **Telefon va laptopni bir xil Wi-Fi ga ulang** ⚠️
2. Telefon browserida konsolda ko'rsatilgan IP manzilni oching
3. "📷 Kamerani Yoqish" tugmasini bosing
4. Kamera ruxsatini bering
5. Real-time detection boshlandi! 🎉

---

## 📱 Interfeys Elementlari

### Boshqaruv Tugmalari:
- **📷 Kamerani Yoqish** - Kamerani yoqadi va detectionni boshlaydi
- **⏹️ To'xtatish** - Kamera va detectionni to'xtatadi
- **🔄 Kamerani Almashtirish** - Old va orqa kamera o'rtasida almashtirish

### Sozlamalar:
- **FPS Slider** - Sekundiga nechta detection (1-30)
  - Kam FPS = kamroq resurs, sekinroq
  - Yuqori FPS = ko'proq resurs, tezroq
  
- **Confidence Threshold** - Minimal ishonchlilik darajasi (0-100%)
  - Kam qiymat = ko'p obyekt, ko'p noto'g'ri
  - Yuqori qiymat = kam obyekt, aniqroq

### Statistika:
- **Aniqlangan obyektlar** - Hozirda ekranda ko'rinayotgan obyektlar soni
- **FPS** - Haqiqiy sekundiga kadrlar
- **Latency** - Server javob vaqti (millisekundlarda)

---

## 🔧 Muammolarni Hal Qilish

### ❌ Serverga ulanib bo'lmayapti

**Sabab:** Telefon va laptop turli Wi-Fi tarmog'ida

**Yechim:**
1. Ikkalasini ham bir xil Wi-Fi ga ulang
2. VPN ishlamayotganligiga ishonch hosil qiling
3. Firewall YOLO serveriga ruxsat berayotganini tekshiring

**Windows Firewall sozlash:**
```
Control Panel → Windows Defender Firewall → Advanced Settings
→ Inbound Rules → New Rule → Port → TCP 5000 → Allow
```

### ❌ Kamera ishlamayapti

**Sabab:** Browser kameraga ruxsat bermagan

**Yechim:**
1. Browser sozlamalaridan kamera ruxsatini tekshiring
2. Chrome: Settings → Privacy → Site Settings → Camera
3. Safari: Settings → Safari → Camera

### ❌ Detection juda sekin

**Yechim:**
1. FPS ni kamaytiring (5-10 ga)
2. Confidence threshold ni oshiring (0.4-0.5)
3. Yengilroq model ishlatish: `yolov8n.pt`

### ❌ Xotira tugab qolmoqda (Memory Error)

**Yechim:**
```bash
# Nano models (eng yengil)
# app.py faylida o'zgartiring:
models = YOLO('yolov8n.pt')
```

---

## 🎓 YOLOv8 Model Variantlari

| Model | Hajmi | Tezligi | Aniqlik | Tavsiya |
|-------|-------|---------|---------|---------|
| YOLOv8n | 6MB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | O'rta kompyuter |
| YOLOv8s | 22MB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Yaxshi balans |
| YOLOv8m | 52MB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Kuchli kompyuter |
| YOLOv8l | 87MB | ⚡⚡ | ⭐⭐⭐⭐⭐ | Juda aniq |
| YOLOv8x | 136MB | ⚡ | ⭐⭐⭐⭐⭐ | Eng aniq |

**Model o'zgartirish:**
```python
# app.py faylida:
model = YOLO('yolov8s.pt')  # yoki boshqa variant
```

---

## 📊 YOLO Aniqlaydigan Obyektlar (80 ta)

COCO dataset asosida:
- **Odamlar va hayvonlar:** person, cat, dog, bird, horse...
- **Transport:** car, bus, truck, bicycle, motorcycle...
- **Uy buyumlari:** chair, table, bed, sofa, tv...
- **Oziq-ovqat:** apple, banana, pizza, cake...
- **Elektronika:** laptop, phone, keyboard, mouse...

To'liq ro'yxatni ko'rish:
```
http://your-server-ip:5000/classes
```

---

## 🛠️ Kengaytirish va Rivojlantirish

### Custom Model Train Qilish

Agar o'zingizning obyektlaringizni aniqlashni xohlasangiz:

```python
from ultralytics import YOLO

# O'z ma'lumotlaringizda train qilish
model = YOLO('yolov8n.pt')
model.train(data='path/to/dataset.yaml', epochs=100)
```

### Video File'dan Detection

```python
# app.py ga qo'shish mumkin
@app.route('/detect_video', methods=['POST'])
def detect_video():
    video_path = request.json['video_path']
    results = model(video_path, stream=True)
    # ...
```

### Telegram Bot Integration

```python
# Telegram bot orqali rasm yuborish va detection olish
from telegram import Bot

@app.route('/telegram_webhook', methods=['POST'])
def telegram_webhook():
    # Telegram'dan rasm qabul qilish
    # Detection qilish
    # Natijani qaytarish
    pass
```

---

## 🔐 Xavfsizlik

### Production uchun tavsiyalar:

1. **HTTPS ishlatish** (SSL sertifikat)
2. **Authentication qo'shish** (API key)
3. **Rate limiting** (ko'p so'rovlarni cheklash)
4. **CORS sozlamalari** (faqat kerakli domenlar)

```python
# Flask-Limiter o'rnatish
pip install flask-limiter

from flask_limiter import Limiter
limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/detect')
@limiter.limit("10 per minute")  # Minutiga 10 ta so'rov
def detect_objects():
    # ...
```

---

## 📈 Performance Optimizatsiya

### 1. Batch Processing
Ko'p rasmlarni bir vaqtda qayta ishlash:
```python
results = model([img1, img2, img3])
```

### 2. GPU Ishlatish
Agar NVIDIA GPU mavjud bo'lsa:
```python
model = YOLO('yolov8n.pt').to('cuda')
```

### 3. TensorRT Optimizatsiya
```bash
yolo export models=yolov8n.pt format=engine device=0
```

---

## 🤝 Yordam va Qo'llab-quvvatlash

### Muammoga duch keldingizmi?

1. **Dokumentatsiyani o'qing:** [Ultralytics Docs](https://docs.ultralytics.com)
2. **GitHub Issues:** [YOLOv8 Issues](https://github.com/ultralytics/ultralytics/issues)
3. **Community:** [Ultralytics Discord](https://discord.gg/ultralytics)

### Foydali Linklar:
- 📚 [YOLO Rasmiy Sayt](https://www.ultralytics.com)
- 🎥 [Video Tutoriallar](https://www.youtube.com/@Ultralytics)
- 📖 [COCO Dataset](https://cocodataset.org)

---

## 📄 Litsenziya

Bu loyiha o'quv maqsadida yaratilgan. YOLOv8 AGPL-3.0 litsenziyasi ostida.

---

## 🎉 Muvaffaqiyatli Ishlatish!

Savol yoki takliflaringiz bo'lsa, bemalol yozing!

**Happy Coding! 🚀**
