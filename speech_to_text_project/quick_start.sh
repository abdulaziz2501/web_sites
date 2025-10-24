#!/bin/bash

# Speech-to-Text - Tez Boshlash Skripti
# Bu skript barcha kerakli amallarni avtomatik bajaradi

echo "╔════════════════════════════════════════════════════════╗"
echo "║   Speech-to-Text: Tez Boshlash                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# 1. Python versiyasini tekshirish
echo "🔍 Python versiyasi tekshirilmoqda..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Python o'rnatilmagan!"
    exit 1
fi

echo "✓ Python topildi"
echo ""

# 2. Virtual environment yaratish
echo "📦 Virtual environment yaratilmoqda..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment yaratildi"
else
    echo "✓ Virtual environment allaqachon mavjud"
fi

echo ""

# 3. Virtual environment'ni faollashtirish
echo "🔄 Virtual environment faollashtirilmoqda..."

# OS turini aniqlash
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi

echo "✓ Virtual environment faollashtirildi"
echo ""

# 4. Kutubxonalarni o'rnatish
echo "📚 Kutubxonalar o'rnatilmoqda..."
echo "(Bu biroz vaqt olishi mumkin...)"
echo ""

pip install --upgrade pip
pip install -r requirements.txt

# PyTorch CPU versiyasi
echo ""
echo "🔥 PyTorch CPU versiyasi o'rnatilmoqda..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "✓ Barcha kutubxonalar o'rnatildi"
echo ""

# 5. Papkalarni yaratish
echo "📁 Papkalar yaratilmoqda..."

mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/train_test_split
mkdir -p models/checkpoints
mkdir -p models/final_model
mkdir -p logs/tensorboard
mkdir -p src

echo "✓ Papkalar yaratildi"
echo ""

# 6. Fayllarni tekshirish
echo "🔍 Konfiguratsiya tekshirilmoqda..."

if [ -f "config.py" ]; then
    echo "✓ config.py topildi"
else
    echo "❌ config.py topilmadi!"
    exit 1
fi

echo ""

# 7. Parquet faylni tekshirish
echo "📂 Parquet fayl tekshirilmoqda..."

PARQUET_FILES=$(ls data/raw/*.parquet 2>/dev/null | wc -l)

if [ "$PARQUET_FILES" -gt 0 ]; then
    echo "✓ Parquet fayl topildi: $(ls data/raw/*.parquet)"
else
    echo "⚠️  Parquet fayl topilmadi!"
    echo ""
    echo "Iltimos, Parquet faylni quyidagi papkaga joylashtiring:"
    echo "  data/raw/"
    echo ""
    echo "Keyin config.py faylda PARQUET_FILE nomini o'zgartiring"
fi

echo ""

# 8. Xulosa
echo "╔════════════════════════════════════════════════════════╗"
echo "║   O'RNATISH TUGADI!                                   ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "Keyingi qadamlar:"
echo ""
echo "1. Parquet faylni data/raw/ ga qo'ying (agar qo'ymagan bo'lsangiz)"
echo "2. config.py faylni sozlang"
echo "3. Ma'lumotlarni qayta ishlang:"
echo "   python data_preprocessing.py"
echo ""
echo "4. Modelni o'rgating:"
echo "   python model_training.py"
echo ""
echo "5. Modelni baholang:"
echo "   python model_evaluation.py"
echo ""
echo "6. Yangi audio fayllarni transkripsiya qiling:"
echo "   python inference.py"
echo ""
echo "Batafsil ma'lumot uchun README.md ni o'qing!"
echo ""
