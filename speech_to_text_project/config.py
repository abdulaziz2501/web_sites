"""
Speech-to-Text Model Konfiguratsiya Fayli
Bu faylda barcha asosiy sozlamalar saqlanadi
"""

import os
from pathlib import Path

# ===== ASOSIY YO'LLAR =====
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_TEST_DIR = DATA_DIR / "train_test_split"

MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
FINAL_MODEL_DIR = MODELS_DIR / "final_model"

# Papkalarni yaratish
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_TEST_DIR, 
                  CHECKPOINTS_DIR, FINAL_MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ===== MODEL SOZLAMALARI =====
class ModelConfig:
    """Model uchun sozlamalar"""
    
    # Qaysi modelni ishlatamiz (CPU uchun optimallashtirilgan)
    # Kichik va tez model tanlaymiz
    MODEL_NAME = "facebook/wav2vec2-base"  # Yoki "openai/whisper-tiny"
    
    # CPU sozlamalari
    USE_CPU = True
    NUM_WORKERS = 2  # CPU core'lar soni (laptop uchun 2-4)
    
    # Training parametrlari
    BATCH_SIZE = 4  # CPU uchun kichik batch size
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 10
    WARMUP_STEPS = 500
    
    # Gradient accumulation (kichik batch size ni qoplash uchun)
    GRADIENT_ACCUMULATION_STEPS = 4
    
    # Mixed precision (CPU uchun o'chirilgan)
    FP16 = False
    
    # Checkpoint saqlash
    SAVE_STEPS = 500
    EVAL_STEPS = 500
    LOGGING_STEPS = 100
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 3


# ===== AUDIO SOZLAMALARI =====
class AudioConfig:
    """Audio ma'lumotlar uchun sozlamalar"""
    
    # Audio parametrlari
    SAMPLE_RATE = 16000  # 16kHz (standart speech recognition uchun)
    MAX_AUDIO_LENGTH = 30  # Maksimal audio uzunligi (soniyalarda)
    
    # Audio normalizatsiya
    NORMALIZE_AUDIO = True
    
    # Augmentation (kerak bo'lsa)
    USE_AUGMENTATION = False
    AUGMENTATION_PROB = 0.5


# ===== MA'LUMOTLAR SOZLAMALARI =====
class DataConfig:
    """Ma'lumotlar bilan ishlash sozlamalari"""
    
    # Parquet fayl nomi (siz yuklaysiz)
    PARQUET_FILE = "your_data.parquet"
    
    # Ma'lumotlar ustunlari (Parquet fayldagi ustun nomlari)
    AUDIO_COLUMN = "audio"  # Audio fayl yo'li yoki bytes
    TEXT_COLUMN = "text"    # Transkripsiya matni
    
    # Train/Test/Validation bo'linmasi
    TRAIN_SIZE = 0.8
    VALID_SIZE = 0.1
    TEST_SIZE = 0.1
    
    # Random seed (takrorlanish uchun)
    RANDOM_SEED = 42
    
    # Maksimal matn uzunligi
    MAX_TEXT_LENGTH = 500


# ===== BAHOLASH SOZLAMALARI =====
class EvalConfig:
    """Model baholash sozlamalari"""
    
    # Metriklar
    METRICS = ["wer", "cer"]  # Word Error Rate, Character Error Rate
    
    # Test batch size
    TEST_BATCH_SIZE = 8


# ===== LOGGING VA MONITORING =====
class LogConfig:
    """Logging sozlamalari"""
    
    LOG_DIR = BASE_DIR / "logs"
    LOG_DIR.mkdir(exist_ok=True)
    
    LOG_FILE = LOG_DIR / "training.log"
    TENSORBOARD_DIR = LOG_DIR / "tensorboard"


# ===== OPTIMIZATSIYA SOZLAMALARI =====
class OptimizationConfig:
    """CPU uchun optimizatsiya"""
    
    # PyTorch sozlamalari
    TORCH_NUM_THREADS = 4  # CPU thread'lar soni
    
    # Memory optimization
    PIN_MEMORY = False  # CPU uchun False
    PREFETCH_FACTOR = 2
    
    # Gradient checkpointing (memory tejash uchun)
    USE_GRADIENT_CHECKPOINTING = True


# ===== EXPORT =====
# Barcha konfiguratsiyalarni eksport qilish
def get_config():
    """Barcha konfiguratsiyalarni dict sifatida qaytaradi"""
    return {
        "model": ModelConfig,
        "audio": AudioConfig,
        "data": DataConfig,
        "eval": EvalConfig,
        "log": LogConfig,
        "optimization": OptimizationConfig
    }


if __name__ == "__main__":
    # Konfiguratsiyani tekshirish
    print("=== Speech-to-Text Konfiguratsiya ===")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Model: {ModelConfig.MODEL_NAME}")
    print(f"CPU Mode: {ModelConfig.USE_CPU}")
    print(f"Batch Size: {ModelConfig.BATCH_SIZE}")
    print(f"Sample Rate: {AudioConfig.SAMPLE_RATE}")
    print("=" * 40)
