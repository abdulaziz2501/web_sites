"""
Ma'lumotlarni tayyorlash va qayta ishlash
Parquet fayldan audio va text ma'lumotlarni o'qish va tayorlash
"""

import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import torch
from datasets import Dataset, DatasetDict
from transformers import Wav2Vec2Processor
import warnings
warnings.filterwarnings('ignore')

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_TEST_DIR,
    ModelConfig, AudioConfig, DataConfig
)


class DataPreprocessor:
    """Ma'lumotlarni qayta ishlash klassi"""
    
    def __init__(self):
        """Initsializatsiya"""
        self.sample_rate = AudioConfig.SAMPLE_RATE
        self.max_audio_length = AudioConfig.MAX_AUDIO_LENGTH
        self.max_text_length = DataConfig.MAX_TEXT_LENGTH
        
        # Processor (tokenizer va feature extractor)
        print(f"Model yuklanmoqda: {ModelConfig.MODEL_NAME}")
        self.processor = Wav2Vec2Processor.from_pretrained(
            ModelConfig.MODEL_NAME
        )
        
    def load_parquet(self, parquet_path):
        """
        Parquet faylni yukash
        
        Args:
            parquet_path: Parquet fayl yo'li
            
        Returns:
            DataFrame: Yuklangan ma'lumotlar
        """
        print(f"\n{'='*50}")
        print(f"Parquet fayl yuklanmoqda: {parquet_path}")
        print(f"{'='*50}")
        
        df = pd.read_parquet(parquet_path)
        
        print(f"✓ Jami qatorlar: {len(df)}")
        print(f"✓ Ustunlar: {list(df.columns)}")
        print(f"\nBirinchi qator ko'rinishi:")
        print(df.head(1))
        
        return df
    
    def load_audio(self, audio_path_or_bytes):
        """
        Audio faylni yuklash va qayta ishlash
        
        Args:
            audio_path_or_bytes: Fayl yo'li yoki bytes
            
        Returns:
            numpy array: Audio signal
        """
        try:
            # Agar fayl yo'li bo'lsa
            if isinstance(audio_path_or_bytes, (str, Path)):
                audio, sr = librosa.load(
                    audio_path_or_bytes, 
                    sr=self.sample_rate,
                    mono=True
                )
            # Agar bytes bo'lsa
            elif isinstance(audio_path_or_bytes, bytes):
                import io
                audio, sr = sf.read(io.BytesIO(audio_path_or_bytes))
                if sr != self.sample_rate:
                    audio = librosa.resample(
                        audio, 
                        orig_sr=sr, 
                        target_sr=self.sample_rate
                    )
            else:
                raise ValueError(f"Noma'lum audio format: {type(audio_path_or_bytes)}")
            
            # Audio normalizatsiya
            if AudioConfig.NORMALIZE_AUDIO:
                audio = librosa.util.normalize(audio)
            
            # Maksimal uzunlikni cheklash
            max_samples = int(self.max_audio_length * self.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            return audio
            
        except Exception as e:
            print(f"⚠ Audio yuklashda xato: {e}")
            return None
    
    def preprocess_text(self, text):
        """
        Matnni tozalash va qayta ishlash
        
        Args:
            text: Asl matn
            
        Returns:
            str: Tozalangan matn
        """
        if not isinstance(text, str):
            return ""
        
        # Bo'sh joylarni tozalash
        text = text.strip()
        
        # Ortiqcha bo'shliqlarni olib tashlash
        text = " ".join(text.split())
        
        # Kichik harflarga o'tkazish (opsional)
        # text = text.lower()
        
        # Maksimal uzunlikni cheklash
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
        
        return text
    
    def process_dataset(self, df):
        """
        Butun datasetni qayta ishlash
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            Dataset: Hugging Face Dataset
        """
        print(f"\n{'='*50}")
        print("Dataset qayta ishlanmoqda...")
        print(f"{'='*50}")
        
        processed_data = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                # Audio yuklash
                audio_data = self.load_audio(row[DataConfig.AUDIO_COLUMN])
                if audio_data is None:
                    continue
                
                # Matnni qayta ishlash
                text = self.preprocess_text(row[DataConfig.TEXT_COLUMN])
                if not text:
                    continue
                
                # Ma'lumotni qo'shish
                processed_data.append({
                    'audio': audio_data,
                    'text': text,
                    'sampling_rate': self.sample_rate
                })
                
            except Exception as e:
                print(f"⚠ {idx}-qatorda xato: {e}")
                continue
        
        print(f"\n✓ Muvaffaqiyatli qayta ishlandi: {len(processed_data)} ta")
        print(f"✗ Xato bo'lgan: {len(df) - len(processed_data)} ta")
        
        # Hugging Face Dataset ga o'tkazish
        dataset = Dataset.from_dict({
            'audio': [x['audio'] for x in processed_data],
            'text': [x['text'] for x in processed_data],
            'sampling_rate': [x['sampling_rate'] for x in processed_data]
        })
        
        return dataset
    
    def prepare_dataset_for_training(self, batch):
        """
        Training uchun batchni tayyorlash
        
        Args:
            batch: Batch ma'lumotlar
            
        Returns:
            dict: Tayyorlangan batch
        """
        # Audio ni encoding qilish
        audio = batch["audio"]
        
        # Feature extraction
        inputs = self.processor(
            audio, 
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Text tokenization
        with self.processor.as_target_processor():
            labels = self.processor(
                batch["text"], 
                return_tensors="pt",
                padding=True
            ).input_ids
        
        batch["input_values"] = inputs.input_values[0]
        batch["labels"] = labels[0]
        
        return batch
    
    def split_dataset(self, dataset):
        """
        Datasetni train/valid/test ga bo'lish
        
        Args:
            dataset: To'liq dataset
            
        Returns:
            DatasetDict: Bo'lingan dataset
        """
        print(f"\n{'='*50}")
        print("Dataset bo'linmoqda...")
        print(f"{'='*50}")
        
        # Train va temp (valid+test) ga bo'lish
        train_test = dataset.train_test_split(
            test_size=1 - DataConfig.TRAIN_SIZE,
            seed=DataConfig.RANDOM_SEED
        )
        
        # Temp ni valid va test ga bo'lish
        valid_size_ratio = DataConfig.VALID_SIZE / (DataConfig.VALID_SIZE + DataConfig.TEST_SIZE)
        valid_test = train_test['test'].train_test_split(
            test_size=1 - valid_size_ratio,
            seed=DataConfig.RANDOM_SEED
        )
        
        # DatasetDict yaratish
        dataset_dict = DatasetDict({
            'train': train_test['train'],
            'validation': valid_test['train'],
            'test': valid_test['test']
        })
        
        print(f"✓ Train: {len(dataset_dict['train'])} samples")
        print(f"✓ Validation: {len(dataset_dict['validation'])} samples")
        print(f"✓ Test: {len(dataset_dict['test'])} samples")
        
        return dataset_dict
    
    def save_dataset(self, dataset_dict, output_dir=TRAIN_TEST_DIR):
        """
        Datasetni saqlash
        
        Args:
            dataset_dict: DatasetDict
            output_dir: Saqlash papkasi
        """
        print(f"\n{'='*50}")
        print("Dataset saqlanmoqda...")
        print(f"{'='*50}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Har bir split ni alohida saqlash
        for split_name, split_dataset in dataset_dict.items():
            split_path = output_dir / split_name
            split_dataset.save_to_disk(str(split_path))
            print(f"✓ {split_name} saqlandi: {split_path}")


def main():
    """Asosiy funksiya - ma'lumotlarni qayta ishlash"""
    
    print("""
    ╔════════════════════════════════════════════════════╗
    ║   Speech-to-Text: Ma'lumotlar qayta ishlash       ║
    ╚════════════════════════════════════════════════════╝
    """)
    
    # Preprocessor yaratish
    preprocessor = DataPreprocessor()
    
    # Parquet faylni yuklash
    parquet_path = RAW_DATA_DIR / DataConfig.PARQUET_FILE
    
    if not parquet_path.exists():
        print(f"⚠ XATO: Parquet fayl topilmadi: {parquet_path}")
        print(f"\nIltimos, Parquet faylni quyidagi papkaga joylashtiring:")
        print(f"  {RAW_DATA_DIR}/")
        print(f"\nYoki config.py faylda PARQUET_FILE nomini o'zgartiring")
        return
    
    df = preprocessor.load_parquet(parquet_path)
    
    # Datasetni qayta ishlash
    dataset = preprocessor.process_dataset(df)
    
    # Train/valid/test ga bo'lish
    dataset_dict = preprocessor.split_dataset(dataset)
    
    # Datasetni saqlash
    preprocessor.save_dataset(dataset_dict)
    
    print(f"\n{'='*50}")
    print("✓ Ma'lumotlar muvaffaqiyatli qayta ishlandi!")
    print(f"{'='*50}")
    print(f"\nKeyingi qadam: python src/model_training.py")


if __name__ == "__main__":
    main()
