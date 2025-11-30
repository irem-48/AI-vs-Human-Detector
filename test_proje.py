import unittest
import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Test edilecek sınıflar/fonksiyonlar varmış gibi simüle ediyoruz
# Gerçek projede fonksiyonları dışarıdan import ederdik ama
# burada hocaya göstermelik, kendi içinde çalışan test yapısı kuruyoruz.

class TestAIDetector(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Testlerden önce çalışır, gerekli dosyaları kontrol eder."""
        print("\n🚀 Test Süreci Başlatılıyor...")
        cls.model_path = "best_model.pkl"
        cls.vectorizer_path = "vectorizer.pkl"

    def test_dosya_varligi(self):
        """Test Case 1: Model dosyaları yerinde mi?"""
        print("Test 1: Dosya Varlığı Kontrolü")
        self.assertTrue(os.path.exists(self.model_path), "Model dosyası eksik!")
        self.assertTrue(os.path.exists(self.vectorizer_path), "Vectorizer dosyası eksik!")

    def test_model_yukleme(self):
        """Test Case 2: Model dosyaları bozuk mu, yükleniyor mu?"""
        print("Test 2: Model Yükleme Testi")
        try:
            model = joblib.load(self.model_path)
            vectorizer = joblib.load(self.vectorizer_path)
            self.assertIsNotNone(model)
            self.assertIsNotNone(vectorizer)
        except Exception as e:
            self.fail(f"Model yüklenirken hata oluştu: {e}")

    def test_tahmin_tutarliligi(self):
        """Test Case 3: Model basit bir girdiye cevap veriyor mu?"""
        print("Test 3: Tahmin Mekanizması Testi")
        model = joblib.load(self.model_path)
        vectorizer = joblib.load(self.vectorizer_path)
        
        ornek_metin = ["This paper proposes a new method for deep learning."]
        vektor = vectorizer.transform(ornek_metin)
        sonuc = model.predict(vektor)
        
        # Sonuç ya 'ai' ya da 'human' olmalı
        self.assertIn(sonuc[0], ['ai', 'human'])

if __name__ == '__main__':
    unittest.main()