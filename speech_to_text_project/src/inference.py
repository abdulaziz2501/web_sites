"""
Inference - Yangi Audio Fayllarni Transkripsiya Qilish
Train qilingan modeldan foydalanib yangi audio fayllarni matnга o'girish
"""

import torch
import librosa
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import FINAL_MODEL_DIR, AudioConfig


class SpeechToTextInference:
    """Speech-to-Text inference klassi"""
    
    def __init__(self, model_path=FINAL_MODEL_DIR):
        """
        Initsializatsiya
        
        Args:
            model_path: Model saqlangan yo'l
        """
        print(f"\n{'='*60}")
        print("Speech-to-Text Inference System")
        print(f"{'='*60}")
        
        # Device
        self.device = torch.device("cpu")
        self.sample_rate = AudioConfig.SAMPLE_RATE
        
        # Model va processor yuklash
        print(f"✓ Model yuklanmoqda: {model_path}")
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Device: {self.device}")
        print("✓ Model tayyor!")
    
    def load_audio(self, audio_path):
        """
        Audio faylni yuklash
        
        Args:
            audio_path: Audio fayl yo'li
            
        Returns:
            numpy array: Audio signal
        """
        print(f"\n📂 Audio yuklanmoqda: {audio_path}")
        
        # Audio yuklash
        audio, sr = librosa.load(
            audio_path,
            sr=self.sample_rate,
            mono=True
        )
        
        # Normalizatsiya
        if AudioConfig.NORMALIZE_AUDIO:
            audio = librosa.util.normalize(audio)
        
        print(f"✓ Audio yuklandi")
        print(f"  • Uzunligi: {len(audio)/self.sample_rate:.2f} soniya")
        print(f"  • Sample rate: {self.sample_rate} Hz")
        
        return audio
    
    def transcribe(self, audio_path):
        """
        Audio faylni transkripsiya qilish
        
        Args:
            audio_path: Audio fayl yo'li
            
        Returns:
            str: Transkripsiya qilingan matn
        """
        # Audio yuklash
        audio = self.load_audio(audio_path)
        
        # Processing
        print("\n🔄 Transkripsiya qilinmoqda...")
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Prediction
        with torch.no_grad():
            logits = self.model(
                inputs.input_values.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device)
            ).logits
        
        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription
    
    def transcribe_batch(self, audio_paths):
        """
        Bir nechta audio faylni transkripsiya qilish
        
        Args:
            audio_paths: Audio fayllar ro'yxati
            
        Returns:
            list: Transkripsiya qilingan matnlar
        """
        print(f"\n{'='*60}")
        print(f"Batch transkripsiya: {len(audio_paths)} ta fayl")
        print(f"{'='*60}")
        
        transcriptions = []
        
        for i, audio_path in enumerate(audio_paths, 1):
            print(f"\n[{i}/{len(audio_paths)}]")
            transcription = self.transcribe(audio_path)
            transcriptions.append(transcription)
            print(f"✓ Natija: {transcription}")
        
        return transcriptions
    
    def transcribe_from_microphone(self, duration=5):
        """
        Mikrofondan audio yozish va transkripsiya qilish
        
        Args:
            duration: Yozish davomiyligi (soniyalarda)
            
        Returns:
            str: Transkripsiya qilingan matn
        """
        import sounddevice as sd
        
        print(f"\n{'='*60}")
        print(f"🎤 Mikrofon yozuvi ({duration} soniya)")
        print(f"{'='*60}")
        print("Gapiring...")
        
        # Audio yozish
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        
        print("✓ Yozuv tugadi")
        
        # Normalizatsiya
        audio = audio.flatten()
        if AudioConfig.NORMALIZE_AUDIO:
            audio = librosa.util.normalize(audio)
        
        # Processing
        print("\n🔄 Transkripsiya qilinmoqda...")
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Prediction
        with torch.no_grad():
            logits = self.model(
                inputs.input_values.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device)
            ).logits
        
        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription


def demo_single_file():
    """Bitta faylni transkripsiya qilish demo"""
    
    # Inference obyekti yaratish
    inference = SpeechToTextInference()
    
    # Audio fayl yo'li
    audio_file = input("\n📂 Audio fayl yo'lini kiriting: ").strip()
    
    if not Path(audio_file).exists():
        print(f"❌ Fayl topilmadi: {audio_file}")
        return
    
    # Transkripsiya
    transcription = inference.transcribe(audio_file)
    
    # Natija
    print(f"\n{'='*60}")
    print("📝 NATIJA:")
    print(f"{'='*60}")
    print(transcription)
    print(f"{'='*60}")


def demo_batch():
    """Bir nechta faylni transkripsiya qilish demo"""
    
    # Inference obyekti yaratish
    inference = SpeechToTextInference()
    
    # Audio fayllar papkasi
    folder = input("\n📂 Audio fayllar papkasini kiriting: ").strip()
    folder_path = Path(folder)
    
    if not folder_path.exists():
        print(f"❌ Papka topilmadi: {folder}")
        return
    
    # Audio fayllarni topish
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(folder_path.glob(f"*{ext}"))
    
    if not audio_files:
        print(f"❌ Audio fayllar topilmadi: {folder}")
        return
    
    print(f"\n✓ Topildi: {len(audio_files)} ta audio fayl")
    
    # Transkripsiya
    transcriptions = inference.transcribe_batch(audio_files)
    
    # Natijalarni saqlash
    output_file = folder_path / "transcriptions.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for audio_file, transcription in zip(audio_files, transcriptions):
            f.write(f"File: {audio_file.name}\n")
            f.write(f"Text: {transcription}\n")
            f.write("-" * 60 + "\n")
    
    print(f"\n✓ Natijalar saqlandi: {output_file}")


def demo_microphone():
    """Mikrofondan transkripsiya qilish demo"""
    
    try:
        import sounddevice
    except ImportError:
        print("❌ sounddevice kutubxonasi o'rnatilmagan")
        print("O'rnatish: pip install sounddevice")
        return
    
    # Inference obyekti yaratish
    inference = SpeechToTextInference()
    
    # Davomiylik
    duration = int(input("\n⏱ Yozuv davomiyligi (soniyalarda): ").strip() or "5")
    
    # Transkripsiya
    transcription = inference.transcribe_from_microphone(duration)
    
    # Natija
    print(f"\n{'='*60}")
    print("📝 NATIJA:")
    print(f"{'='*60}")
    print(transcription)
    print(f"{'='*60}")


def main():
    """Asosiy menyu"""
    
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║   Speech-to-Text: Inference (Transkripsiya)           ║
    ╚════════════════════════════════════════════════════════╝
    
    Nima qilmoqchisiz?
    
    1. Bitta audio faylni transkripsiya qilish
    2. Bir nechta faylni transkripsiya qilish (batch)
    3. Mikrofondan yozish va transkripsiya qilish
    4. Chiqish
    """)
    
    choice = input("Tanlovingiz (1-4): ").strip()
    
    if choice == "1":
        demo_single_file()
    elif choice == "2":
        demo_batch()
    elif choice == "3":
        demo_microphone()
    elif choice == "4":
        print("\n👋 Xayr!")
        return
    else:
        print("\n❌ Noto'g'ri tanlov!")


if __name__ == "__main__":
    main()
