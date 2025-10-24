# Speech-to-Text Model - Tez Boshlash Yo'riqnomasi

Ushbu yo'riqnoma sizga 5 daqiqada loyihani boshlash imkonini beradi.

---

## 🚀 3 Qadam - Boshlash

### 1️⃣ O'rnatish (2 daqiqa)

**Windows:**
```bash
quick_start.bat
```

**Linux/Mac:**
```bash
chmod +x quick_start.sh
./quick_start.sh
```

### 2️⃣ Parquet Faylni Joylashtirish (1 daqiqa)

```bash
# Sizning parquet faylingizni data/raw/ ga ko'chiring
data/raw/sizning_fayl.parquet
```

`config.py` faylni oching va o'zgartiring:
```python
class DataConfig:
    PARQUET_FILE = "sizning_fayl.parquet"
    AUDIO_COLUMN = "audio"  # Parquet dagi ustun nomi
    TEXT_COLUMN = "text"    # Parquet dagi ustun nomi
```

### 3️⃣ Ishga Tushirish (2 daqiqa setup)

```bash
# Virtual environment faollashtirish
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# Ma'lumotlarni qayta ishlash
python data_preprocessing.py
```

---

## 📊 To'liq Jarayon (4 qadam)

### Qadam 1: Ma'lumotlarni Tayyorlash
```bash
python data_preprocessing.py
```
⏱ Vaqt: 10-60 daqiqa

### Qadam 2: Model O'qitish
```bash
python model_training.py
```
⏱ Vaqt: 2-10 soat

### Qadam 3: Model Baholash
```bash
python model_evaluation.py
```
⏱ Vaqt: 5-30 daqiqa

### Qadam 4: Transkripsiya
```bash
python inference.py
```
⏱ Vaqt: Darhol

---

## 🎯 Umumiy Sozlamalar

`config.py` faylda o'zgartirishingiz mumkin:

```python
# Model
MODEL_NAME = "facebook/wav2vec2-base"  # Yoki "openai/whisper-tiny"

# CPU sozlamalari
BATCH_SIZE = 4  # Kichik laptop uchun 2
NUM_EPOCHS = 10  # Training davrlari

# Audio
SAMPLE_RATE = 16000  # 16kHz
MAX_AUDIO_LENGTH = 30  # 30 soniya
```

---

## 🆘 Tez Yechimlar

### Xato: "Module not found"
```bash
pip install -r requirements.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Xato: "Parquet file not found"
```bash
# Faylni to'g'ri joyda ekanini tekshiring
ls data/raw/
# config.py da nomini to'g'rilang
```

### Memory xatosi
```python
# config.py da
BATCH_SIZE = 1  # Kichikroq qiling
```

---

## 📚 Dokumentatsiya

- **To'liq o'rnatish:** `INSTALLATION.md`
- **Batafsil yo'riqnoma:** `README.md`
- **Konfiguratsiya:** `config.py`

---

## 📁 Loyiha Strukturasi

```
.
├── config.py              # Sozlamalar
├── requirements.txt       # Kutubxonalar
├── data_preprocessing.py  # Ma'lumotlarni tayyorlash
├── model_training.py      # Model o'qitish
├── model_evaluation.py    # Baholash
├── inference.py           # Transkripsiya
├── data/
│   └── raw/              # Parquet fayl bu yerda
├── models/               # Saqlangan modellar
└── logs/                 # Training logs
```

---

## ✅ Tez Tekshirish

```bash
# 1. Python versiyasi
python --version  # 3.8+ kerak

# 2. Virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Kutubxonalar
python -c "import torch; print(torch.__version__)"

# 4. Parquet fayl
python -c "import pandas as pd; from config import RAW_DATA_DIR, DataConfig; df = pd.read_parquet(RAW_DATA_DIR / DataConfig.PARQUET_FILE); print(f'{len(df)} qator')"
```

---

**Tayyor! Ishni boshlang! 🚀**
