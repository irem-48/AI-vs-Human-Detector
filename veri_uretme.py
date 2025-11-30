import google.generativeai as genai
import pandas as pd
import time
import os

# --- AYARLAR ---
# ⚠️ DİKKAT: Yeni API anahtarını buraya yapıştır
API_KEY = "AIzaSyAnGj3Op3Rz2CItGEEPG3fsHfxNwM7E30g"  
INPUT_FILE = "human_data.csv"
OUTPUT_FILE = "ai_data.csv"

# En yüksek limitlere sahip ve hızlı model (Experimental)
MODEL_NAME = "gemini-2.0-flash"
BATCH_SIZE = 10  # Tek istekte 10 veri işler (Kotayı korur)

# API'yi yapılandır
genai.configure(api_key=API_KEY)

# Güvenlik ayarları (Modelin gereksiz yere veriyi reddetmemesi için)
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model = genai.GenerativeModel(MODEL_NAME, safety_settings=safety_settings)

def generate_ai_data_batch():
    print(f"🚀 Toplu İşlem Başlıyor... Model: {MODEL_NAME} | Paket Boyutu: {BATCH_SIZE}")
    
    if not os.path.exists(INPUT_FILE):
        print("❌ Hata: human_data.csv dosyası bulunamadı!")
        return

    # CSV Okuma (Encoding hatası almamak için utf-8 eklendi)
    try:
        df_human = pd.read_csv(INPUT_FILE, encoding='utf-8')
    except UnicodeDecodeError:
        df_human = pd.read_csv(INPUT_FILE, encoding='latin1') # Yedek encoding
    
    # Kaldığımız yeri belirle
    baslangic_index = 0
    if os.path.exists(OUTPUT_FILE):
        try:
            df_ai = pd.read_csv(OUTPUT_FILE)
            baslangic_index = len(df_ai)
            print(f"🔄 Önceki dosyadan devam ediliyor. İşlenen veri sayısı: {baslangic_index}")
        except pd.errors.EmptyDataError:
            pass

    toplam_veri = len(df_human)
    print(f"📂 Toplam kaynak veri: {toplam_veri}")
    
    # Batch döngüsü
    for i in range(baslangic_index, toplam_veri, BATCH_SIZE):
        
        # 1. Adım: Mevcut paketi al
        batch_df = df_human.iloc[i : i + BATCH_SIZE]
        texts_to_process = batch_df['text'].tolist()
        
        if not texts_to_process:
            break

        print(f"\n📦 Paket işleniyor: {i} - {i + len(texts_to_process)} arası... (Toplam: {toplam_veri})")

        try:
            # 2. Adım: Prompt oluştur
            joined_texts = ""
            for idx, text in enumerate(texts_to_process):
                joined_texts += f"--- METİN {idx+1} ---\n{text}\n\n"

            prompt = f"""
            Aşağıda sana {len(texts_to_process)} adet metin veriyorum.
            Bu metinlerin HER BİRİNİ, bir AI asistanı gibi daha basit, açıklayıcı ve net bir dille yeniden yaz (rewrite/paraphrase).
            
            KURALLAR:
            1. Tam olarak {len(texts_to_process)} adet çıktı üret.
            2. Çıktıların sırasını asla değiştirme.
            3. Her bir cevabın arasına SADECE "|||" işaretini koy.
            4. Giriş veya sonuç cümlesi yazma. Sadece metinler ve ayraçlar.

            METİNLER:
            {joined_texts}
            """
            
            # API Çağrısı
            response = model.generate_content(prompt)
            
            # 3. Adım: Cevabı işle
            if response.text:
                generated_list = response.text.split("|||")
                cleaned_list = [t.strip() for t in generated_list if t.strip()]

                # Sayı kontrolü
                if len(cleaned_list) == len(texts_to_process):
                    yeni_veriler = [{"text": ai_text, "label": "ai"} for ai_text in cleaned_list]
                    
                    # Kaydet
                    temp_df = pd.DataFrame(yeni_veriler)
                    header_durumu = not os.path.exists(OUTPUT_FILE)
                    temp_df.to_csv(OUTPUT_FILE, mode='a', header=header_durumu, index=False, encoding='utf-8')
                    print(f"✅ {len(yeni_veriler)} veri kaydedildi.")
                else:
                    print(f"⚠️ UYARI: {len(texts_to_process)} gönderildi, {len(cleaned_list)} alındı. Paket atlanıyor.")
            
            # Başarılı olursa kısa bekle
            time.sleep(2) 

        except Exception as e:
            hata_mesaji = str(e)
            print(f"❌ Hata: {hata_mesaji}")
            
            if "429" in hata_mesaji or "quota" in hata_mesaji.lower():
                print("⏳ Kota limiti! 60 sn bekleniyor...")
                time.sleep(60)
                # Retry mekanizması e
# Dosyanın EN SONUNA eklenecek kısım:
if __name__ == "__main__":
    generate_ai_data_batch()