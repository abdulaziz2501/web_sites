"""
Speech-to-Text Model O'qitish (CPU Optimized)
CPU da samarali ishlash uchun optimallashtirilgan
"""

import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_from_disk
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Union
import warnings
warnings.filterwarnings('ignore')

from config import (
    TRAIN_TEST_DIR, CHECKPOINTS_DIR, FINAL_MODEL_DIR,
    ModelConfig, OptimizationConfig, LogConfig
)


# CPU uchun PyTorch sozlamalari
torch.set_num_threads(OptimizationConfig.TORCH_NUM_THREADS)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator - batch yaratish uchun
    Padding qo'shadi va batch ni to'g'ri formatda qaytaradi
    """
    
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Audio va text ni ajratish
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # Audio features ni padding qilish
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Labels ni padding qilish
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )
        
        # Padding token'larni -100 ga o'zgartirish (loss hisoblashda e'tibor berilmaydi)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        batch["labels"] = labels
        
        return batch


class SpeechToTextTrainer:
    """Model o'qitish klassi"""
    
    def __init__(self):
        """Initsializatsiya"""
        print(f"\n{'='*60}")
        print("Speech-to-Text Trainer - CPU Mode")
        print(f"{'='*60}")
        
        # Device (CPU)
        self.device = torch.device("cpu")
        print(f"✓ Device: {self.device}")
        
        # Processor yuklash
        print(f"✓ Processor yuklanmoqda: {ModelConfig.MODEL_NAME}")
        self.processor = Wav2Vec2Processor.from_pretrained(
            ModelConfig.MODEL_NAME
        )
        
        # Model yuklash
        print(f"✓ Model yuklanmoqda: {ModelConfig.MODEL_NAME}")
        self.model = Wav2Vec2ForCTC.from_pretrained(
            ModelConfig.MODEL_NAME,
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        
        # Model ni CPU ga ko'chirish
        self.model.to(self.device)
        
        # Gradient checkpointing (memory tejash uchun)
        if OptimizationConfig.USE_GRADIENT_CHECKPOINTING:
            self.model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing yoqildi")
        
        # Model parametrlari
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
    def load_datasets(self):
        """
        Train, validation, test datasetlarni yuklash
        
        Returns:
            tuple: (train_dataset, valid_dataset, test_dataset)
        """
        print(f"\n{'='*60}")
        print("Datasetlar yuklanmoqda...")
        print(f"{'='*60}")
        
        train_dataset = load_from_disk(str(TRAIN_TEST_DIR / "train"))
        valid_dataset = load_from_disk(str(TRAIN_TEST_DIR / "validation"))
        test_dataset = load_from_disk(str(TRAIN_TEST_DIR / "test"))
        
        print(f"✓ Train: {len(train_dataset)} samples")
        print(f"✓ Validation: {len(valid_dataset)} samples")
        print(f"✓ Test: {len(test_dataset)} samples")
        
        return train_dataset, valid_dataset, test_dataset
    
    def prepare_dataset(self, batch):
        """
        Datasetni training uchun tayyorlash
        
        Args:
            batch: Batch ma'lumotlar
            
        Returns:
            dict: Tayyorlangan batch
        """
        # Audio ni processing qilish
        audio = batch["audio"]
        
        # Agar numpy array bo'lsa
        if isinstance(audio, np.ndarray):
            # Feature extraction
            batch["input_values"] = self.processor(
                audio, 
                sampling_rate=batch["sampling_rate"],
            ).input_values[0]
        
        # Text tokenization
        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["text"]).input_ids
        
        return batch
    
    def compute_metrics(self, pred):
        """
        Model metrikalarini hisoblash (WER - Word Error Rate)
        
        Args:
            pred: Prediction natijasi
            
        Returns:
            dict: Metrikalar
        """
        from jiwer import wer
        
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        
        # Padding token'larni olib tashlash
        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id
        
        # Decode qilish
        pred_str = self.processor.batch_decode(pred_ids)
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)
        
        # WER hisoblash
        wer_score = wer(label_str, pred_str)
        
        return {"wer": wer_score}
    
    def create_trainer(self, train_dataset, valid_dataset):
        """
        Trainer yaratish
        
        Args:
            train_dataset: Train dataseti
            valid_dataset: Validation dataseti
            
        Returns:
            Trainer: Hugging Face Trainer obyekti
        """
        print(f"\n{'='*60}")
        print("Trainer sozlanmoqda...")
        print(f"{'='*60}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(CHECKPOINTS_DIR),
            
            # CPU sozlamalari
            use_cpu=True,
            no_cuda=True,
            
            # Batch sizes
            per_device_train_batch_size=ModelConfig.BATCH_SIZE,
            per_device_eval_batch_size=ModelConfig.BATCH_SIZE,
            gradient_accumulation_steps=ModelConfig.GRADIENT_ACCUMULATION_STEPS,
            
            # Epochs va learning rate
            num_train_epochs=ModelConfig.NUM_EPOCHS,
            learning_rate=ModelConfig.LEARNING_RATE,
            warmup_steps=ModelConfig.WARMUP_STEPS,
            
            # Evaluation
            eval_strategy="steps",
            eval_steps=ModelConfig.EVAL_STEPS,
            
            # Saving
            save_strategy="steps",
            save_steps=ModelConfig.SAVE_STEPS,
            save_total_limit=3,  # Faqat oxirgi 3 ta checkpoint saqlanadi
            
            # Logging
            logging_dir=str(LogConfig.TENSORBOARD_DIR),
            logging_steps=ModelConfig.LOGGING_STEPS,
            
            # Optimizatsiya
            fp16=False,  # CPU uchun mixed precision ishlamaydi
            dataloader_num_workers=ModelConfig.NUM_WORKERS,
            
            # Best model saqlash
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,  # WER past bo'lsa yaxshi
            
            # Report
            report_to=["tensorboard"],
            
            # Seed
            seed=42,
        )
        
        # Data collator
        data_collator = DataCollatorCTCWithPadding(
            processor=self.processor,
            padding=True
        )
        
        # Trainer yaratish
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=ModelConfig.EARLY_STOPPING_PATIENCE
                )
            ]
        )
        
        print("✓ Trainer tayyor")
        
        return trainer
    
    def train(self):
        """
        Modelni o'qitish (asosiy funksiya)
        """
        print(f"\n{'='*60}")
        print("TRAINING BOSHLANDI")
        print(f"{'='*60}")
        
        # Datasetlarni yuklash
        train_dataset, valid_dataset, test_dataset = self.load_datasets()
        
        # Datasetlarni tayyorlash
        print("\nDatasetlar tayyorlanmoqda...")
        train_dataset = train_dataset.map(
            self.prepare_dataset,
            remove_columns=train_dataset.column_names,
            num_proc=1
        )
        
        valid_dataset = valid_dataset.map(
            self.prepare_dataset,
            remove_columns=valid_dataset.column_names,
            num_proc=1
        )
        
        print("✓ Datasetlar tayyor")
        
        # Trainer yaratish
        trainer = self.create_trainer(train_dataset, valid_dataset)
        
        # Training
        print(f"\n{'='*60}")
        print("Model o'qitilmoqda... (Bu uzoq vaqt olishi mumkin)")
        print(f"{'='*60}")
        
        trainer.train()
        
        print(f"\n{'='*60}")
        print("✓ TRAINING TUGADI!")
        print(f"{'='*60}")
        
        # Modelni saqlash
        self.save_model(trainer)
        
        return trainer
    
    def save_model(self, trainer):
        """
        Tayyor modelni saqlash
        
        Args:
            trainer: Trainer obyekti
        """
        print(f"\n{'='*60}")
        print("Model saqlanmoqda...")
        print(f"{'='*60}")
        
        # Model va processor ni saqlash
        trainer.save_model(str(FINAL_MODEL_DIR))
        self.processor.save_pretrained(str(FINAL_MODEL_DIR))
        
        print(f"✓ Model saqlandi: {FINAL_MODEL_DIR}")
        print(f"\n{'='*60}")
        print("Keyingi qadam: python src/model_evaluation.py")
        print(f"{'='*60}")


def main():
    """Asosiy funksiya"""
    
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║   Speech-to-Text: Model O'qitish (CPU Optimized)      ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    # Trainer yaratish
    trainer_obj = SpeechToTextTrainer()
    
    # Training
    trainer_obj.train()
    
    print("\n✓ Jarayon muvaffaqiyatli tugadi!")


if __name__ == "__main__":
    main()
