"""
Model Baholash va Test
Train qilingan modelni test qilish va metrikalarni ko'rsatish
"""

import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_from_disk
from jiwer import wer, cer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import TRAIN_TEST_DIR, FINAL_MODEL_DIR, ModelConfig


class ModelEvaluator:
    """Model baholash klassi"""
    
    def __init__(self, model_path=FINAL_MODEL_DIR):
        """
        Initsializatsiya
        
        Args:
            model_path: Model saqlangan yo'l
        """
        print(f"\n{'='*60}")
        print("Model Evaluator - Baholash tizimi")
        print(f"{'='*60}")
        
        # Device
        self.device = torch.device("cpu")
        print(f"✓ Device: {self.device}")
        
        # Model va processor yuklash
        print(f"✓ Model yuklanmoqda: {model_path}")
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()  # Evaluation mode
        
        print("✓ Model yuklandi va tayyor")
    
    def load_test_dataset(self):
        """
        Test datasetni yuklash
        
        Returns:
            Dataset: Test dataseti
        """
        print(f"\n{'='*60}")
        print("Test dataset yuklanmoqda...")
        print(f"{'='*60}")
        
        test_dataset = load_from_disk(str(TRAIN_TEST_DIR / "test"))
        print(f"✓ Test samples: {len(test_dataset)}")
        
        return test_dataset
    
    def transcribe_audio(self, audio_array, sampling_rate):
        """
        Bitta audio faylni transkripsiya qilish
        
        Args:
            audio_array: Audio signal (numpy array)
            sampling_rate: Sampling rate
            
        Returns:
            str: Transkripsiya qilingan matn
        """
        # Audio ni processing qilish
        inputs = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Model bilan prediction
        with torch.no_grad():
            logits = self.model(
                inputs.input_values.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device)
            ).logits
        
        # Logits dan token IDs ni olish
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Decode qilish
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription
    
    def evaluate_dataset(self, dataset):
        """
        Butun datasetni baholash
        
        Args:
            dataset: Test dataseti
            
        Returns:
            dict: Metrikalar
        """
        print(f"\n{'='*60}")
        print("Dataset baholanmoqda...")
        print(f"{'='*60}")
        
        predictions = []
        references = []
        
        for sample in tqdm(dataset, desc="Evaluating"):
            # Audio transkripsiya qilish
            pred_text = self.transcribe_audio(
                sample["audio"],
                sample["sampling_rate"]
            )
            
            predictions.append(pred_text)
            references.append(sample["text"])
        
        # Metrikalarni hisoblash
        wer_score = wer(references, predictions)
        cer_score = cer(references, predictions)
        
        results = {
            "wer": wer_score,
            "cer": cer_score,
            "num_samples": len(dataset)
        }
        
        return results, predictions, references
    
    def display_results(self, results, predictions=None, references=None, num_examples=5):
        """
        Natijalarni ko'rsatish
        
        Args:
            results: Metrikalar
            predictions: Prediction'lar (opsional)
            references: Reference'lar (opsional)
            num_examples: Nechta misol ko'rsatish
        """
        print(f"\n{'='*60}")
        print("BAHOLASH NATIJALARI")
        print(f"{'='*60}")
        
        print(f"\n📊 Metrikalar:")
        print(f"  • WER (Word Error Rate):      {results['wer']:.4f} ({results['wer']*100:.2f}%)")
        print(f"  • CER (Character Error Rate): {results['cer']:.4f} ({results['cer']*100:.2f}%)")
        print(f"  • Test samples:               {results['num_samples']}")
        
        # Natijalarni baholash
        if results['wer'] < 0.1:
            grade = "A'LO ✨"
        elif results['wer'] < 0.2:
            grade = "YAXSHI ✓"
        elif results['wer'] < 0.3:
            grade = "O'RTACHA ≈"
        else:
            grade = "YOMON ✗"
        
        print(f"\n🎯 Model bahosi: {grade}")
        
        # Misollar ko'rsatish
        if predictions and references:
            print(f"\n{'='*60}")
            print(f"MISOLLAR (birinchi {num_examples} ta):")
            print(f"{'='*60}")
            
            for i in range(min(num_examples, len(predictions))):
                print(f"\n📝 Misol #{i+1}:")
                print(f"  Asl matn:        {references[i]}")
                print(f"  Model javobi:    {predictions[i]}")
                
                # WER ni hisoblash (bitta misol uchun)
                sample_wer = wer([references[i]], [predictions[i]])
                print(f"  WER:             {sample_wer:.4f} ({sample_wer*100:.2f}%)")
    
    def save_results(self, results, predictions, references, output_file="evaluation_results.txt"):
        """
        Natijalarni faylga saqlash
        
        Args:
            results: Metrikalar
            predictions: Prediction'lar
            references: Reference'lar
            output_file: Fayl nomi
        """
        from pathlib import Path
        
        output_path = Path(FINAL_MODEL_DIR) / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("SPEECH-TO-TEXT MODEL EVALUATION RESULTS\n")
            f.write("="*60 + "\n\n")
            
            f.write("📊 METRIKALAR:\n")
            f.write(f"  WER: {results['wer']:.4f} ({results['wer']*100:.2f}%)\n")
            f.write(f"  CER: {results['cer']:.4f} ({results['cer']*100:.2f}%)\n")
            f.write(f"  Samples: {results['num_samples']}\n\n")
            
            f.write("="*60 + "\n")
            f.write("BARCHA NATIJALAR:\n")
            f.write("="*60 + "\n\n")
            
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                f.write(f"Sample #{i+1}:\n")
                f.write(f"  Reference:   {ref}\n")
                f.write(f"  Prediction:  {pred}\n")
                f.write(f"  WER:         {wer([ref], [pred]):.4f}\n")
                f.write("\n")
        
        print(f"\n✓ Natijalar saqlandi: {output_path}")


def main():
    """Asosiy funksiya"""
    
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║   Speech-to-Text: Model Baholash                      ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    # Evaluator yaratish
    evaluator = ModelEvaluator()
    
    # Test datasetni yuklash
    test_dataset = evaluator.load_test_dataset()
    
    # Dataset baholash
    results, predictions, references = evaluator.evaluate_dataset(test_dataset)
    
    # Natijalarni ko'rsatish
    evaluator.display_results(results, predictions, references, num_examples=10)
    
    # Natijalarni saqlash
    evaluator.save_results(results, predictions, references)
    
    print(f"\n{'='*60}")
    print("✓ Baholash tugadi!")
    print(f"{'='*60}")
    print("\nKeyingi qadam: python src/inference.py")


if __name__ == "__main__":
    main()
