#!/bin/bash

echo "===================================="
echo "🚀 YOLOv8 Detection Server"
echo "===================================="
echo ""

# Virtual environment mavjudligini tekshirish
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment topilmadi!"
    echo "📦 Yangi environment yaratilmoqda..."
    python3 -m venv venv
    echo "✅ Virtual environment yaratildi"
    echo ""
fi

# Virtual environment aktivlashtirish
echo "🔄 Virtual environment aktivlashtirilmoqda..."
source venv/bin/activate

# Dependency'larni tekshirish
echo "🔍 Dependency'lar tekshirilmoqda..."
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Dependency'lar o'rnatilmoqda..."
    echo "⏳ Bu bir necha daqiqa davom etishi mumkin..."
    pip install -r requirements.txt
    echo "✅ O'rnatish yakunlandi"
    echo ""
fi

# Serverni ishga tushirish
echo ""
echo "===================================="
echo "▶️  Server ishga tushirilmoqda..."
echo "===================================="
echo ""
python app.py

# Xatolik bo'lsa, xabar berish
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Server xato bilan to'xtadi!"
    read -p "Davom etish uchun Enter bosing..."
fi
