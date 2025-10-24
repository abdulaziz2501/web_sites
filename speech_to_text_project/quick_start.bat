@echo off
REM Speech-to-Text - Tez Boshlash Skripti (Windows)

echo ╔════════════════════════════════════════════════════════╗
echo ║   Speech-to-Text: Tez Boshlash (Windows)             ║
echo ╚════════════════════════════════════════════════════════╝
echo.

REM 1. Python versiyasini tekshirish
echo 🔍 Python versiyasi tekshirilmoqda...
python --version

if %errorlevel% neq 0 (
    echo ❌ Python o'rnatilmagan!
    echo.
    echo Python 3.8+ ni quyidagi havoladan yuklab oling:
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✓ Python topildi
echo.

REM 2. Virtual environment yaratish
echo 📦 Virtual environment yaratilmoqda...

if not exist "venv" (
    python -m venv venv
    echo ✓ Virtual environment yaratildi
) else (
    echo ✓ Virtual environment allaqachon mavjud
)

echo.

REM 3. Virtual environment'ni faollashtirish
echo 🔄 Virtual environment faollashtirilmoqda...
call venv\Scripts\activate.bat
echo ✓ Virtual environment faollashtirildi
echo.

REM 4. Pip yangilash
echo 📦 pip yangilanmoqda...
python -m pip install --upgrade pip
echo.

REM 5. Kutubxonalarni o'rnatish
echo 📚 Kutubxonalar o'rnatilmoqda...
echo (Bu biroz vaqt olishi mumkin...)
echo.

pip install -r requirements.txt

REM PyTorch CPU versiyasi
echo.
echo 🔥 PyTorch CPU versiyasi o'rnatilmoqda...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo ✓ Barcha kutubxonalar o'rnatildi
echo.

REM 6. Papkalarni yaratish
echo 📁 Papkalar yaratilmoqda...

if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\train_test_split" mkdir data\train_test_split
if not exist "models\checkpoints" mkdir models\checkpoints
if not exist "models\final_model" mkdir models\final_model
if not exist "logs\tensorboard" mkdir logs\tensorboard
if not exist "src" mkdir src

echo ✓ Papkalar yaratildi
echo.

REM 7. Fayllarni tekshirish
echo 🔍 Konfiguratsiya tekshirilmoqda...

if exist "config.py" (
    echo ✓ config.py topildi
) else (
    echo ❌ config.py topilmadi!
    pause
    exit /b 1
)

echo.

REM 8. Parquet faylni tekshirish
echo 📂 Parquet fayl tekshirilmoqda...

dir /b data\raw\*.parquet >nul 2>&1

if %errorlevel% equ 0 (
    echo ✓ Parquet fayl topildi
    dir data\raw\*.parquet
) else (
    echo ⚠️  Parquet fayl topilmadi!
    echo.
    echo Iltimos, Parquet faylni quyidagi papkaga joylashtiring:
    echo   data\raw\
    echo.
    echo Keyin config.py faylda PARQUET_FILE nomini o'zgartiring
)

echo.

REM 9. Xulosa
echo ╔════════════════════════════════════════════════════════╗
echo ║   O'RNATISH TUGADI!                                   ║
echo ╚════════════════════════════════════════════════════════╝
echo.
echo Keyingi qadamlar:
echo.
echo 1. Parquet faylni data\raw\ ga qo'ying (agar qo'ymagan bo'lsangiz)
echo 2. config.py faylni sozlang
echo 3. Ma'lumotlarni qayta ishlang:
echo    python data_preprocessing.py
echo.
echo 4. Modelni o'rgating:
echo    python model_training.py
echo.
echo 5. Modelni baholang:
echo    python model_evaluation.py
echo.
echo 6. Yangi audio fayllarni transkripsiya qiling:
echo    python inference.py
echo.
echo Batafsil ma'lumot uchun README.md ni o'qing!
echo.

pause
