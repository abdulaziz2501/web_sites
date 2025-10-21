@echo off
chcp 65001 >nul
echo ====================================
echo 🚀 YOLOv8 Detection Server
echo ====================================
echo.

REM Virtual environment mavjudligini tekshirish
if not exist "venv\" (
    echo ❌ Virtual environment topilmadi!
    echo 📦 Yangi environment yaratilmoqda...
    python -m venv venv
    echo ✅ Virtual environment yaratildi
    echo.
)

REM Virtual environment aktivlashtirish
echo 🔄 Virtual environment aktivlashtirilmoqda...
call venv\Scripts\activate.bat

REM Dependency'larni tekshirish
echo 🔍 Dependency'lar tekshirilmoqda...
python -c "import flask" 2>nul
if errorlevel 1 (
    echo 📦 Dependency'lar o'rnatilmoqda...
    echo ⏳ Bu bir necha daqiqa davom etishi mumkin...
    pip install -r requirements.txt
    echo ✅ O'rnatish yakunlandi
    echo.
)

REM Serverni ishga tushirish
echo.
echo ====================================
echo ▶️  Server ishga tushirilmoqda...
echo ====================================
echo.
python app.py

REM Xatolik bo'lsa, console ochiq qoladi
if errorlevel 1 (
    echo.
    echo ❌ Server xato bilan to'xtadi!
    pause
)
