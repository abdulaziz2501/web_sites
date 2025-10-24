# Speech-to-Text Model (CPU Optimized)

Audio fayllarni matnга o'giradigan Speech-to-Text modeli. CPU da ishlash uchun optimallashtirilgan.

## 📋 Mundarija

- [Xususiyatlar](#xususiyatlar)
- [Texnologiyalar](#texnologiyalar)
- [O'rnatish](#ornatish)
- [Loyiha Strukturasi](#loyiha-strukturasi)
- [Ishlatish Bo'yicha Yo'riqnoma](#ishlatish-boyicha-yoriqnoma)
- [Konfiguratsiya](#konfiguratsiya)
- [Troubleshooting](#troubleshooting)

---

## ✨ Xususiyatlar

- ✅ **CPU Optimized** - Laptop/PCда tez ishlaydi
- ✅ **Parquet Support** - Parquet fayllardan ma'lumot yuklash
- ✅ **Pre-trained Models** - Wav2Vec2 yoki Whisper modellaridan foydalanish
- ✅ **Easy Training** - Sodda va tushunarli training jarayoni
- ✅ **Batch Processing** - Bir nechta faylni birdaniga qayta ishlash
- ✅ **Real-time** - Mikrofondan to'g'ridan-to'g'ri transkripsiya
- ✅ **Evaluation Metrics** - WER va CER metrikalar
- ✅ **Uzbek Language Ready** - O'zbek tilida ishlatish uchun tayyor

---

## 🛠 Texnologiyalar

- **PyTorch** (CPU versiyasi) - Model training
- **Hugging Face Transformers** - Pre-trained modellar
- **Librosa** - Audio processing
- **Pandas** - Ma'lumotlar bilan ishlash
- **Wav2Vec2 / Whisper** - Base modellar

---

## 📦 O'rnatish

### 1. Python versiyasini tekshirish

```bash
python --version
# Python 3.8+ kerak
```

### 2. Virtual environment yaratish (tavsiya etiladi)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Kerakli kutubxonalarni o'rnatish

```bash
# Birinchi: requirements.txt fayldan o'rnatish
pip install -r requirements.txt

# PyTorch CPU versiyasini alohida o'rnatish (MUHIM!)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Parquet faylni joylashtirish

Sizning Parquet faylingizni `data/raw/` papkasiga qo'ying:

```bash
data/raw/your_data.parquet
```

**Muhim:** `config.py` faylda `PARQUET_FILE` nomini o'z faylingiz nomiga o'zgartiring!

---

## 📁 Loyiha Strukturasi

```
speech_to_text_project/
│
├── config.py                    # Barcha sozlamalar
├── requirements.txt             # Python kutubxonalari
├── README.md                    # Ushbu fayl
│
├── data/
│   ├── raw/                     # Asl Parquet fayl
│   ├── processed/               # Qayta ishlangan ma'lumotlar
│   └── train_test_split/        # Train/validation/test
│
├── models/
│   ├── checkpoints/             # Training checkpointlar
│   └── final_model/             # Tayyor model
│
├── src/                         # Asosiy kod fayllari
│   ├── data_preprocessing.py   # Ma'lumotlarni tayyorlash
│   ├── model_training.py       # Model o'rgatish
│   ├── model_evaluation.py     # Model baholash
│   └── inference.py            # Yangi fayllarni transkripsiya qilish
│
└── logs/                        # Training logs
    └── tensorboard/             # TensorBoard fayllari
```

---

## 🚀 Ishlatish Bo'yicha Yo'riqnoma

### BOSQICH 1: Parquet Faylni Ko'rish va Tayyorlash

Birinchi, sizning Parquet faylingiz qanday strukturaga ega ekanini tekshiring:

```python
import pandas as pd

# Parquet faylni yuklash
df = pd.read_parquet("data/raw/your_data.parquet")

# Strukturani ko'rish
print(df.head())
print(df.columns)
```

**Kerakli ustunlar:**
- `audio` - Audio fayl yo'li yoki audio bytes
- `text` - Transkripsiya matni (to'g'ri yozilgan)

Agar ustun nomlari boshqacha bo'lsa, `config.py` da o'zgartiring:

```python
# config.py faylda
class DataConfig:
    AUDIO_COLUMN = "sizning_audio_ustun_nomi"
    TEXT_COLUMN = "sizning_text_ustun_nomi"
```

### BOSQICH 2: Ma'lumotlarni Qayta Ishlash

```bash
python data_preprocessing.py
```

Bu jarayon:
- Parquet faylni o'qiydi
- Audio fayllarni yuklaydi va qayta ishlaydi
- Matnlarni tozalaydi
- Train/Validation/Test ga bo'ladi (80%/10%/10%)
- Ma'lumotlarni `data/train_test_split/` ga saqlaydi

**Kutilgan vaqt:** 10-60 daqiqa (ma'lumotlar hajmiga qarab)

### BOSQICH 3: Modelni O'rgatish

```bash
python model_training.py
```

Bu jarayon:
- Pre-trained modelni yuklaydi (Wav2Vec2 yoki Whisper)
- CPU da training qiladi
- Har bir epoch'da validation qiladi
- Eng yaxshi modelni saqlaydi
- Checkpointlarni yaratadi

**Kutilgan vaqt:** 2-10 soat (ma'lumotlar hajmiga va epoch soniga qarab)

**Training jarayonini kuzatish:**

```bash
# TensorBoard orqali
tensorboard --logdir logs/tensorboard
```

Brauzerda `http://localhost:6006` ni oching.

### BOSQICH 4: Modelni Baholash

```bash
python model_evaluation.py
```

Bu:
- Test datasetda model ishlashini tekshiradi
- WER (Word Error Rate) va CER (Character Error Rate) hisoblaydi
- Natijalarni ko'rsatadi va saqlaydi

**Yaxshi natija:**
- WER < 10% = A'lo
- WER < 20% = Yaxshi
- WER < 30% = O'rtacha

### BOSQICH 5: Yangi Audio Fayllarni Transkripsiya Qilish

```bash
python inference.py
```

**3 ta rejim:**

1. **Bitta fayl** - Bir audio faylni transkripsiya qilish
2. **Batch** - Bir nechta faylni birdaniga
3. **Mikrofon** - Jonli ovoz yozish va transkripsiya

---

## ⚙️ Konfiguratsiya

`config.py` faylda barcha sozlamalarni o'zgartirishingiz mumkin:

### Model Sozlamalari

```python
class ModelConfig:
    MODEL_NAME = "facebook/wav2vec2-base"  # Yoki "openai/whisper-tiny"
    BATCH_SIZE = 4                          # Kichik laptop uchun 2-4
    NUM_EPOCHS = 10                         # O'rgatish davrlari
    LEARNING_RATE = 3e-4                    # Learning rate
```

### Audio Sozlamalari

```python
class AudioConfig:
    SAMPLE_RATE = 16000                     # 16kHz (standart)
    MAX_AUDIO_LENGTH = 30                   # Maksimal 30 soniya
    NORMALIZE_AUDIO = True                  # Normalizatsiya
```

---

## 🔧 Troubleshooting

### 1. PyTorch CPU versiyasi o'rnatilmagan

**Xato:**
```
RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False
```

**Yechim:**
```bash
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Parquet fayl topilmadi

**Xato:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/your_data.parquet'
```

**Yechim:**
- Parquet faylni `data/raw/` ga qo'ying
- `config.py` da `PARQUET_FILE` nomini to'g'rilang

### 3. Memory xatosi

**Xato:**
```
RuntimeError: [enforce fail at alloc_cpu.cpp:75] . DefaultCPUAllocator: can't allocate memory
```

**Yechim:**
- `config.py` da `BATCH_SIZE` ni kamaytiring (masalan, 2 yoki 1)
- `GRADIENT_ACCUMULATION_STEPS` ni oshiring
- Kichikroq model tanlang (`whisper-tiny`)

### 4. Audio yuklashda xato

**Xato:**
```
LibsndfileError: Error opening 'file.mp3': File contains data in an unknown format
```

**Yechim:**
```bash
# ffmpeg o'rnatish kerak
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# Windows: https://ffmpeg.org/download.html dan yuklab oling
```

### 5. Training juda sekin

**Yechim:**
- Kichikroq model ishlating (`wav2vec2-base` o'rniga `wav2vec2-small`)
- `NUM_EPOCHS` ni kamaytiring
- Ma'lumotlar hajmini kamaytiring (faqat bir qismini ishlating)

---

## 📊 Model Taqqoslash

| Model | Hajmi | Tezlik (CPU) | Aniqlik | Tavsiya |
|-------|-------|--------------|---------|---------|
| wav2vec2-small | ~95MB | Tez | Yaxshi | ✅ Boshlang'ich |
| wav2vec2-base | ~360MB | O'rtacha | A'lo | ✅ Tavsiya etiladi |
| whisper-tiny | ~150MB | Tez | O'rtacha | ⚠️ Ingliz tili uchun |
| whisper-base | ~290MB | O'rtacha | Yaxshi | ✅ Ko'p tilli |

---

## 💡 Maslahatlar

1. **Kichik modeldan boshlang** - `wav2vec2-base` yoki `whisper-tiny`
2. **Ma'lumotlar sifati muhim** - Shovqinsiz, aniq audio fayllar
3. **Normalizatsiya** - Har doim audio normalizatsiya qiling
4. **Early stopping** - Overfitting'dan qochish uchun
5. **Checkpoint'lar** - Training jarayonida saqlang
6. **Validation** - Har doim validation set bilan test qiling

---

## 📝 Qo'shimcha

### Yangi til qo'shish

Agar o'zbek yoki boshqa til uchun model o'rgatmoqchi bo'lsangiz:

1. O'zbek tilidagi audio+text dataseti tayyorlang
2. Base modelni fine-tune qiling
3. `config.py` da til sozlamalarini o'zgartiring

### Model export qilish

Train qilingan modelni boshqa formatga export qilish:

```python
# ONNX formatga
# Bu kodni inference.py ga qo'shing
```

---

## 🤝 Qo'llab-quvvatlash

Savol yoki muammo bo'lsa:

1. Birinchi `config.py` ni tekshiring
2. Log fayllarni ko'rib chiqing (`logs/training.log`)
3. TensorBoard'da training jarayonini kuzating

---

## 📜 Litsenziya

Bu loyiha o'quv maqsadida yaratilgan va erkin foydalanish uchun.

---

## ✅ Keyingi Qadamlar

1. ✅ Parquet faylni joylashtiring
2. ✅ Ma'lumotlarni qayta ishlang (`python data_preprocessing.py`)
3. ✅ Modelni o'rgating (`python model_training.py`)
4. ✅ Modelni baholang (`python model_evaluation.py`)
5. ✅ Inference qiling (`python inference.py`)

---

**Omad tilaymiz! 🚀**
