# Speech-to-Text Model - O'rnatish Yo'riqnomasi

Bu yo'riqnoma sizga Speech-to-Text modelini noldan qanday o'rnatish va ishga tushirishni ko'rsatadi.

---

## 📋 Tizim Talablari

### Minimal Talablar:
- **OS:** Windows 10/11, Ubuntu 20.04+, macOS 10.15+
- **RAM:** 8 GB (16 GB tavsiya etiladi)
- **Disk:** 10 GB bo'sh joy
- **Python:** 3.8, 3.9, 3.10, yoki 3.11
- **Internet:** Modellarni yuklash uchun

### CPU Spetsifikatsiyalari:
- **Minimal:** Intel Core i5 / AMD Ryzen 5
- **Tavsiya etiladi:** Intel Core i7 / AMD Ryzen 7
- **Core'lar:** 4+ core (ko'proq - yaxshiroq)

---

## 🔧 1. Python O'rnatish

### Windows:

1. [Python.org](https://www.python.org/downloads/) ga kiring
2. Python 3.8+ ni yuklab oling
3. O'rnatishda **"Add Python to PATH"** ni belgilang
4. Terminalda tekshiring:
```bash
python --version
```

### Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### macOS:

```bash
# Homebrew orqali
brew install python@3.10
```

---

## 📦 2. Loyiha Fayllarini Joylashtirish

### Variantlar:

**Variant A: Barcha fayllar bitta papkada**

```bash
# Windows
mkdir C:\speech-to-text
cd C:\speech-to-text

# Linux/Mac
mkdir ~/speech-to-text
cd ~/speech-to-text
```

Barcha yuklangan fayllarni bu papkaga ko'chiring.

**Variant B: Loyiha strukturasini yaratish**

Qo'lda yaratish:

```
speech-to-text/
├── config.py
├── requirements.txt
├── README.md
├── data_preprocessing.py
├── model_training.py
├── model_evaluation.py
├── inference.py
├── data/
│   └── raw/          # Parquet fayl bu yerda
├── models/
└── logs/
```

---

## 🚀 3. Tez O'rnatish (Avtomatik)

### Windows:

1. `quick_start.bat` faylini ikki marta bosing
2. Yoki CMD/PowerShell'da:
```bash
quick_start.bat
```

### Linux/Mac:

```bash
chmod +x quick_start.sh
./quick_start.sh
```

Bu skript avtomatik ravishda:
- Virtual environment yaratadi
- Kutubxonalarni o'rnatadi
- Papkalarni sozlaydi
- Barcha kerakli fayllarni tekshiradi

---

## 🛠 4. Qo'lda O'rnatish (Batafsil)

Agar avtomatik o'rnatish ishlamasa, quyidagi qadamlarni bajaring:

### 4.1. Virtual Environment Yaratish

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Virtual environment faollashtirilganini tekshirish:
```bash
which python  # Linux/Mac
where python  # Windows
```

### 4.2. Pip Yangilash

```bash
python -m pip install --upgrade pip
```

### 4.3. Requirements O'rnatish

```bash
pip install -r requirements.txt
```

**Agar xato bo'lsa:**

```bash
# Har bir kutubxonani alohida o'rnatish
pip install pandas numpy pyarrow
pip install librosa soundfile audioread
pip install transformers datasets tokenizers
pip install jiwer evaluate
pip install matplotlib seaborn tqdm
pip install scikit-learn pyyaml
```

### 4.4. PyTorch CPU Versiyasi

**MUHIM:** PyTorch CPU versiyasini alohida o'rnatish kerak!

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Tekshirish:
```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Natija `CUDA available: False` bo'lishi kerak.

### 4.5. Papkalarni Yaratish

```bash
# Windows
mkdir data\raw data\processed data\train_test_split
mkdir models\checkpoints models\final_model
mkdir logs\tensorboard

# Linux/Mac
mkdir -p data/{raw,processed,train_test_split}
mkdir -p models/{checkpoints,final_model}
mkdir -p logs/tensorboard
```

---

## 📂 5. Parquet Faylni Joylashtirish

1. Sizning Parquet faylingizni `data/raw/` papkasiga ko'chiring

2. `config.py` faylni oching va quyidagi qismni o'zgartiring:

```python
class DataConfig:
    PARQUET_FILE = "sizning_fayl_nomi.parquet"  # O'z faylingiz nomi
    AUDIO_COLUMN = "audio"     # Parquet dagi audio ustun nomi
    TEXT_COLUMN = "text"       # Parquet dagi text ustun nomi
```

3. Tekshirish:

```python
python -c "from config import RAW_DATA_DIR, DataConfig; print(RAW_DATA_DIR / DataConfig.PARQUET_FILE)"
```

---

## 🧪 6. Test va Tekshirish

### Python Import Testi:

```python
python -c "import torch, transformers, librosa, pandas; print('✓ Barcha kutubxonalar ishlayapti')"
```

### Konfiguratsiya Testi:

```bash
python config.py
```

Natijada sizning sozlamalaringiz ko'rsatiladi.

### Parquet Fayl Testi:

```python
python -c "import pandas as pd; from config import RAW_DATA_DIR, DataConfig; df = pd.read_parquet(RAW_DATA_DIR / DataConfig.PARQUET_FILE); print(f'✓ Parquet yuklandi: {len(df)} qator')"
```

---

## 🎯 7. Birinchi Ishga Tushirish

Hammasi tayyor bo'lgach, quyidagi tartibda boshlang:

### 7.1. Ma'lumotlarni Tayyorlash

```bash
python data_preprocessing.py
```

Bu 10-60 daqiqa olishi mumkin.

### 7.2. Modelni O'rgatish

```bash
python model_training.py
```

Bu 2-10 soat olishi mumkin. Training jarayonida:

```bash
# Boshqa terminalda TensorBoard'ni ochish
tensorboard --logdir logs/tensorboard
```

### 7.3. Modelni Baholash

```bash
python model_evaluation.py
```

### 7.4. Yangi Audio'larni Transkripsiya Qilish

```bash
python inference.py
```

---

## ❌ Muammolar va Yechimlar

### 1. "Python command not found"

**Yechim:**
```bash
# Python PATH ga qo'shilganini tekshiring
# Windows: PATH environment variable'ga qo'shing
# Linux/Mac: .bashrc yoki .zshrc ga qo'shing
export PATH="/usr/local/bin/python3:$PATH"
```

### 2. "No module named 'torch'"

**Yechim:**
```bash
# Virtual environment faollashtirilganini tekshiring
# Aks holda:
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# PyTorch qayta o'rnatish
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. "CUDA out of memory"

**Yechim:**
Bu xato CPU rejimida bo'lmasligi kerak. Agar bo'lsa:
```python
# config.py da
class ModelConfig:
    USE_CPU = True
    BATCH_SIZE = 2  # Kichikroq qiling
```

### 4. "RuntimeError: Attempting to deserialize object on CUDA"

**Yechim:**
PyTorch GPU versiyasi o'rnatilgan. CPU versiyasini o'rnating:
```bash
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5. "FileNotFoundError: Parquet file not found"

**Yechim:**
```bash
# Parquet fayl to'g'ri joyda ekanini tekshiring
ls data/raw/  # Linux/Mac
dir data\raw\  # Windows

# config.py da fayl nomini to'g'rilang
```

### 6. Memory xatosi (Training paytida)

**Yechim:**
```python
# config.py da
class ModelConfig:
    BATCH_SIZE = 1  # Eng kichik qiymat
    GRADIENT_ACCUMULATION_STEPS = 8  # Oshiring
    
    # Kichikroq model tanlang
    MODEL_NAME = "openai/whisper-tiny"
```

### 7. "Connection timeout" (Model yuklashda)

**Yechim:**
```bash
# Proxy sozlash (kerak bo'lsa)
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port

# Yoki model'ni qo'lda yuklab cache'ga qo'ying
```

---

## 📚 Qo'shimcha Resurslar

### Hugging Face Models:
- [Wav2Vec2 Models](https://huggingface.co/models?filter=wav2vec2)
- [Whisper Models](https://huggingface.co/models?filter=whisper)

### Documentation:
- [PyTorch CPU](https://pytorch.org/get-started/locally/)
- [Transformers](https://huggingface.co/docs/transformers)
- [Librosa](https://librosa.org/doc/latest/)

---

## ✅ O'rnatish Checklist

Quyidagi barcha bandlarni belgilang:

- [ ] Python 3.8+ o'rnatilgan
- [ ] Virtual environment yaratilgan va faollashtirilgan
- [ ] Barcha requirements o'rnatilgan
- [ ] PyTorch CPU versiyasi o'rnatilgan
- [ ] Papka strukturasi yaratilgan
- [ ] Parquet fayl `data/raw/` da
- [ ] `config.py` sozlangan
- [ ] Import testlari muvaffaqiyatli
- [ ] Parquet fayl yuklanadi

Hammasi belgilansa, tayyor! 🎉

---

## 🆘 Yordam

Agar muammo hal bo'lmasa:

1. **Log fayllarni ko'ring:** `logs/training.log`
2. **Xato xabarini to'liq ko'chiring**
3. **Tizim ma'lumotlarini yig'ing:**
```bash
python --version
pip list
```

---

**Muvaffaqiyatlar! 🚀**
