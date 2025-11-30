# YAZILIM ŞARTNAMESİ (SOFTWARE REQUIREMENTS SPECIFICATION - SRS)

## Human vs AI Text Detector Projesi

---

**Belge Versiyonu:** 1.0  
**Hazırlanma Tarihi:** 2024  
**Proje Başlangıç Tarihi:** 1 Kasım 2024  
**Proje Teslim Tarihi:** 19 Aralık 2024  
**Dil:** Türkçe

---

## 1. GİRİŞ

### 1.1. Projenin Tanımı ve Amacı

Bu proje, akademik ve genel metinlerin **Yapay Zeka (AI)** tarafından mı yoksa **İnsan** tarafından mı yazıldığını tespit etmek amacıyla geliştirilmiş bir makine öğrenmesi tabanlı sınıflandırma sistemidir. Proje, özellikle akademik ortamlarda artan AI destekli içerik üretiminin tespiti için kritik bir araç olarak tasarlanmıştır.

**Temel Amaçlar:**
- Akademik metinlerin orijinalliğini değerlendirmek
- Büyük Dil Modelleri (LLM) tarafından üretilen içerikleri tespit etmek
- İnsan yazımı metinler ile AI üretimi metinler arasındaki farklılıkları makine öğrenmesi algoritmaları ile analiz etmek
- Kullanıcı dostu bir web arayüzü üzerinden gerçek zamanlı metin analizi sağlamak

### 1.2. Belgenin Kapsamı

Bu belge, Human vs AI Text Detector projesinin tüm fonksiyonel ve teknik gereksinimlerini, sistem mimarisini, kullanılan teknolojileri ve zaman planlamasını detaylı olarak açıklamaktadır.

### 1.3. Tanımlar ve Kısaltmalar

- **AI (Artificial Intelligence):** Yapay Zeka
- **LLM (Large Language Model):** Büyük Dil Modelleri (GPT, Gemini, vb.)
- **ML (Machine Learning):** Makine Öğrenmesi
- **TF-IDF:** Term Frequency-Inverse Document Frequency (Terim Sıklığı-Ters Belge Sıklığı)
- **SRS:** Software Requirements Specification (Yazılım Şartnamesi)
- **API:** Application Programming Interface (Uygulama Programlama Arayüzü)

---

## 2. GENEL BAKIŞ

### 2.1. Sistem Mimarisi

Proje, aşağıdaki ana bileşenlerden oluşmaktadır:

1. **Veri Toplama Modülü** (`veri_toplama.py`)
   - ArXiv akademik veritabanından insan yazımı metinlerin toplanması

2. **Veri Üretim Modülü** (`veri_uretme.py`)
   - Google Gemini API kullanılarak AI üretimi metinlerin oluşturulması

3. **Model Eğitim Modülü** (`model_egitimi.py`)
   - Makine öğrenmesi modellerinin eğitimi ve en iyi modelin seçilmesi

4. **Web Uygulaması** (`app.py`)
   - Streamlit tabanlı kullanıcı arayüzü ve gerçek zamanlı tahmin servisi

5. **Test Modülü** (`test_proje.py`)
   - Sistemin doğruluğunu ve güvenilirliğini test eden birim testleri

### 2.2. Teknoloji Yığını

- **Programlama Dili:** Python 3.x
- **Makine Öğrenmesi Kütüphanesi:** Scikit-learn
- **Web Framework:** Streamlit
- **Veri İşleme:** Pandas
- **Model Persistence:** Joblib
- **AI API:** Google Gemini API (gemini-2.0-flash)
- **Veri Kaynağı:** ArXiv API

---

## 3. FONKSİYONEL GEREKSİNİMLER

### 3.1. Kullanıcı Metin Girişi (FR-001)

**Gereksinim:** Sistem, kullanıcıların analiz edilmek üzere metin girişi yapabilmesine olanak sağlamalıdır.

**Açıklama:**
- Kullanıcı, web arayüzü üzerinden bir metin alanına (text area) metin yapıştırabilir veya yazabilir
- Metin girişi minimum 1 karakter olmalıdır
- Sistem, boş metin girişlerini tespit edip uyarı mesajı göstermelidir

**Öncelik:** Yüksek  
**Kabul Kriterleri:**
- Metin girişi başarılı bir şekilde alınabilmeli
- Boş girişler için uyarı mesajı gösterilmeli

### 3.2. Metin Analizi ve Tahmin (FR-002)

**Gereksinim:** Sistem, girilen metni analiz ederek AI veya İnsan yazımı olup olmadığını tespit etmelidir.

**Açıklama:**
- Sistem, TF-IDF vektörizasyonu kullanarak metni sayısallaştırmalı
- Eğitilmiş makine öğrenmesi modeli ile tahmin yapmalı
- Tahmin sonucu "ai" veya "human" etiketi ile döndürülmeli

**Öncelik:** Yüksek  
**Kabul Kriterleri:**
- Metin başarılı bir şekilde analiz edilmeli
- Tahmin sonucu doğru formatta döndürülmeli

### 3.3. Olasılık Skoru Gösterimi (FR-003)

**Gereksinim:** Sistem, tahmin sonucunu yüzde cinsinden olasılık skoru ile göstermelidir.

**Açıklama:**
- Sistem, her sınıf (AI ve İnsan) için olasılık skorunu hesaplamalı
- Olasılık skorları yüzde formatında gösterilmeli (örn: %85 AI, %15 İnsan)
- Görsel olarak progress bar (ilerleme çubuğu) ile desteklenmeli
- Sonuç, hangi sınıfın daha yüksek olasılığa sahip olduğuna göre vurgulanmalı

**Öncelik:** Yüksek  
**Kabul Kriterleri:**
- Olasılık skorları doğru hesaplanmalı
- Görsel gösterim kullanıcı dostu olmalı
- Sonuç net bir şekilde anlaşılabilir olmalı

**Örnek Çıktı:**
- 🤖 Yapay Zeka Skoru: %85.3
- 👤 İnsan Skoru: %14.7
- Sonuç: **YAPAY ZEKA** (%85.3)

### 3.4. Veri Seti Gereksinimleri (FR-004)

**Gereksinim:** Sistem, en az 6000 adet etiketli veri ile eğitilmelidir.

**Açıklama:**
- Veri seti, insan yazımı ve AI üretimi metinlerden oluşmalıdır
- Veri seti dengeli (balanced) olmalı veya dengesizlik durumunda uygun teknikler uygulanmalıdır
- Veri seti, akademik metinlerden oluşmalıdır (ArXiv kaynaklı)

**Öncelik:** Yüksek  
**Kabul Kriterleri:**
- Toplam veri sayısı minimum 6000 olmalı
- Veri seti CSV formatında saklanmalı (`human_data.csv`, `ai_data.csv`)
- Veri kalitesi kontrol edilmeli (boş, tekrar eden veriler temizlenmeli)

### 3.5. Model Eğitimi ve Seçimi (FR-005)

**Gereksinim:** Sistem, birden fazla makine öğrenmesi algoritmasını test edip en iyi performans gösteren modeli seçmelidir.

**Açıklama:**
- Sistem, aşağıdaki algoritmaları test etmelidir:
  - Naive Bayes (MultinomialNB)
  - Random Forest Classifier
  - Logistic Regression
- Model seçimi, test seti üzerindeki doğruluk (accuracy) skoruna göre yapılmalıdır
- En iyi model, `best_model.pkl` dosyasına kaydedilmelidir
- TF-IDF vektörizer, `vectorizer.pkl` dosyasına kaydedilmelidir

**Öncelik:** Yüksek  
**Kabul Kriterleri:**
- Tüm modeller başarılı bir şekilde eğitilmeli
- En iyi model otomatik olarak seçilmeli
- Model dosyaları doğru formatta kaydedilmeli

### 3.6. Veri Toplama (FR-006)

**Gereksinim:** Sistem, ArXiv akademik veritabanından insan yazımı metinler toplayabilmelidir.

**Açıklama:**
- ArXiv API kullanılarak akademik makale özetleri çekilmelidir
- Toplanan veriler "human" etiketi ile işaretlenmelidir
- Veriler CSV formatında saklanmalıdır

**Öncelik:** Orta  
**Kabul Kriterleri:**
- ArXiv API'den veri başarılı bir şekilde çekilmeli
- Veriler doğru formatta kaydedilmeli

### 3.7. AI Veri Üretimi (FR-007)

**Gereksinim:** Sistem, Google Gemini API kullanarak AI üretimi metinler oluşturabilmelidir.

**Açıklama:**
- Google Gemini API (gemini-2.0-flash modeli) kullanılmalıdır
- İnsan yazımı metinler, AI tarafından yeniden yazılarak (paraphrase) AI veri seti oluşturulmalıdır
- Batch işleme (toplu işleme) desteği olmalıdır
- API kota limitleri göz önünde bulundurulmalıdır

**Öncelik:** Orta  
**Kabul Kriterleri:**
- Gemini API başarılı bir şekilde çağrılmalı
- Üretilen metinler "ai" etiketi ile işaretlenmeli
- Veriler CSV formatında saklanmalıdır

---

## 4. TEKNİK GEREKSİNİMLER

### 4.1. Yazılım Gereksinimleri

**Programlama Dili:** Python 3.7 veya üzeri

**Gerekli Python Kütüphaneleri:**
- streamlit
- scikit-learn
- pandas
- joblib
- google-generativeai
- arxiv

Detaylı kütüphane listesi ve versiyonları için `requirements.txt` dosyasına bakınız (Ek A).

### 4.2. Donanım Gereksinimleri

- **Minimum RAM:** 4 GB
- **Disk Alanı:** 500 MB (veri setleri ve model dosyaları dahil)
- **İnternet Bağlantısı:** API çağrıları için gerekli

### 4.3. Dış Servisler

- **Google Gemini API:** AI veri üretimi için API anahtarı gereklidir
- **ArXiv API:** Akademik veri toplama için gerekli (ücretsiz, API anahtarı gerektirmez)

---

## 5. KAPSAM

### 5.1. Proje Kapsamı İçinde Olanlar

- Akademik metinlerin AI/İnsan tespiti
- Web tabanlı kullanıcı arayüzü
- Makine öğrenmesi modeli eğitimi ve tahmin
- Olasılık skoru gösterimi
- Veri toplama ve üretim araçları
- Model performans değerlendirmesi

### 5.2. Proje Kapsamı Dışında Olanlar

- Gerçek zamanlı çoklu kullanıcı desteği (production-ready ölçeklenebilirlik)
- Veritabanı entegrasyonu
- Kullanıcı kimlik doğrulama ve yetkilendirme
- API servisi olarak dağıtım
- Mobil uygulama geliştirme
- Çoklu dil desteği (sadece İngilizce metinler desteklenir)

---

## 6. ZAMAN PLANLAMASI

### 6.1. Proje Zaman Çizelgesi

| Faz | Açıklama | Başlangıç Tarihi | Bitiş Tarihi | Süre |
|-----|----------|------------------|--------------|------|
| **Faz 1** | Veri Toplama ve Üretimi | 1 Kasım 2024 | 15 Kasım 2024 | 2 hafta |
| **Faz 2** | Model Eğitimi ve Optimizasyon | 16 Kasım 2024 | 5 Aralık 2024 | 3 hafta |
| **Faz 3** | Web Uygulaması Geliştirme | 6 Aralık 2024 | 15 Aralık 2024 | 1.5 hafta |
| **Faz 4** | Test ve Dokümantasyon | 16 Aralık 2024 | 19 Aralık 2024 | 4 gün |

**Toplam Süre:** 7 hafta (1 Kasım - 19 Aralık 2024)

### 6.2. Milestone'lar

- **Milestone 1 (15 Kasım):** Veri seti hazır (minimum 6000 veri)
- **Milestone 2 (5 Aralık):** Model eğitimi tamamlandı ve en iyi model seçildi
- **Milestone 3 (15 Aralık):** Web uygulaması tamamlandı ve test edildi
- **Milestone 4 (19 Aralık):** Proje teslimi

---

## 7. KALİTE GEREKSİNİMLERİ

### 7.1. Performans Gereksinimleri

- Metin analizi süresi: Maksimum 5 saniye (ortalama metin uzunluğu için)
- Model yükleme süresi: Maksimum 3 saniye (ilk yükleme)
- Web arayüzü yanıt süresi: Maksimum 2 saniye

### 7.2. Güvenilirlik Gereksinimleri

- Model doğruluğu: Minimum %75 accuracy (test seti üzerinde)
- Sistem hatası oranı: Maksimum %1
- Veri işleme başarı oranı: Minimum %95

### 7.3. Kullanılabilirlik Gereksinimleri

- Kullanıcı arayüzü sezgisel ve kullanımı kolay olmalı
- Hata mesajları açıklayıcı olmalı
- Sonuçlar görsel olarak anlaşılır şekilde sunulmalı

---

## 8. TEST GEREKSİNİMLERİ

### 8.1. Birim Testler

- Model dosyalarının varlığı kontrolü
- Model yükleme testi
- Tahmin mekanizması testi
- Veri işleme fonksiyonları testi

### 8.2. Entegrasyon Testleri

- Veri toplama ve üretim modüllerinin entegrasyonu
- Model eğitimi ve web uygulaması entegrasyonu
- End-to-end (uçtan uca) test senaryoları

### 8.3. Kullanıcı Kabul Testleri

- Kullanıcı senaryoları test edilmeli
- Farklı metin uzunlukları ve içerikleri test edilmeli
- Hata durumları test edilmeli

---

## 9. GÜVENLİK GEREKSİNİMLERİ

### 9.1. Veri Güvenliği

- API anahtarları kod içinde hardcode edilmemeli (ortam değişkenleri kullanılmalı)
- Kullanıcı girdileri sanitize edilmeli
- Hassas veriler loglanmamalı

### 9.2. API Güvenliği

- Gemini API anahtarı güvenli bir şekilde saklanmalı
- API çağrıları rate limiting ile korunmalı
- Hata mesajlarında hassas bilgiler açığa çıkarılmamalı

---

## 10. BAKIM VE GÜNCELLEME

### 10.1. Model Güncellemeleri

- Yeni verilerle model yeniden eğitilebilir olmalı
- Model versiyonlama sistemi olmalı
- Performans metrikleri takip edilmeli

### 10.2. Kod Bakımı

- Kod yorumları ve dokümantasyon güncel tutulmalı
- Versiyon kontrol sistemi (Git) kullanılmalı
- Kod standartlarına uyulmalı (PEP 8)

---

## 11. EKLER

### Ek A: Kullanılan Kütüphaneler Listesi (requirements.txt)

```
streamlit>=1.28.0
scikit-learn>=1.3.0
pandas>=2.0.0
joblib>=1.3.0
google-generativeai>=0.3.0
arxiv>=2.1.0
```

**Not:** Bu dosya proje kök dizininde `requirements.txt` olarak saklanmalıdır.

### Ek B: Proje Dosya Yapısı

```
AI_Dedektoru/
├── app.py                 # Streamlit web uygulaması
├── model_egitimi.py       # Model eğitim modülü
├── veri_toplama.py        # ArXiv veri toplama modülü
├── veri_uretme.py         # Gemini API ile AI veri üretimi
├── model_kontrol.py       # API model kontrolü
├── test_proje.py          # Birim testleri
├── human_data.csv         # İnsan yazımı veri seti
├── ai_data.csv            # AI üretimi veri seti
├── best_model.pkl         # Eğitilmiş en iyi model
├── vectorizer.pkl         # TF-IDF vektörizer
└── requirements.txt       # Python bağımlılıkları
```

### Ek C: Kullanım Senaryoları

**Senaryo 1: Metin Analizi**
1. Kullanıcı web arayüzünü açar
2. Analiz edilecek metni metin alanına yapıştırır
3. "Analiz Et" butonuna tıklar
4. Sistem metni analiz eder ve sonuçları gösterir

**Senaryo 2: Model Eğitimi**
1. Geliştirici `model_egitimi.py` dosyasını çalıştırır
2. Sistem veri setlerini yükler
3. Birden fazla model eğitilir
4. En iyi model seçilir ve kaydedilir

---

## 12. ONAY

Bu belge, proje ekibi ve paydaşlar tarafından gözden geçirilmiş ve onaylanmıştır.

**Hazırlayan:** [İsim]  
**Tarih:** [Tarih]  
**Onaylayan:** [İsim/Ünvan]  
**Tarih:** [Tarih]

---

**Belge Sonu**

