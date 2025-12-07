import streamlit as st
import joblib
import pandas as pd

# --- SAYFA YAPISI ---
st.set_page_config(
    page_title="AI Code/Text Detector",
    page_icon="🕵️‍♂️",
    layout="centered"
)

# --- BAŞLIK VE AÇIKLAMA ---
st.title("🕵️‍♂️ Gelişmiş AI - İnsan Metin Dedektörü")
st.markdown("""
Bu proje, girilen metnin **Yapay Zeka (AI)** tarafından mı yoksa **Gerçek Bir İnsan** tarafından mı yazıldığını 
Makine Öğrenmesi (ML) algoritmaları kullanarak tespit eder.
""")
# ARAYÜZE EKLENEN TASARIM ÖGESİ
st.info("Not: Bu sistem eğitim amaçlı geliştirilmiştir.")
st.divider()

# --- MODEL YÜKLEME ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_model()

# --- ARAYÜZ ---
if model is None:
    st.error("🚨 HATA: Model dosyaları bulunamadı! Lütfen önce 'model_egitimi.py' dosyasını çalıştırın.")
else:
    # Metin Giriş Alanı
    user_input = st.text_area("Analiz edilecek metni buraya yapıştırın:", height=200, placeholder="Metni buraya girin...")

    if st.button("🔍 Analiz Et", use_container_width=True):
        if not user_input.strip():
            st.warning("⚠️ Lütfen boş bırakmayınız.")
        else:
            with st.spinner('Analiz ediliyor...'):
                # 1. Metni Sayısallaştır
                text_vector = vectorizer.transform([user_input])
                
                # 2. Tahmin Yap
                prediction = model.predict(text_vector)[0]
                probabilities = model.predict_proba(text_vector)[0]
                
                # Olasılıkları Belirle (Sınıf sırasına göre)
                classes = model.classes_
                if classes[0] == "ai":
                    ai_score = probabilities[0]
                    human_score = probabilities[1]
                else:
                    ai_score = probabilities[1]
                    human_score = probabilities[0]

                # 3. Sonuçları Göster
                st.subheader("📊 Analiz Sonucu")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🤖 Yapay Zeka Skoru", f"%{ai_score*100:.1f}")
                    st.progress(ai_score, text="AI Olasılığı")
                
                with col2:
                    st.metric("👤 İnsan Skoru", f"%{human_score*100:.1f}")
                    st.progress(human_score, text="İnsan Olasılığı")

                st.divider()
                
                # Karar Mesajı
                if ai_score > human_score:
                    st.error(f"Sonuç: **YAPAY ZEKA** (%{ai_score*100:.1f})")
                    st.info("Bu metin yüksek ihtimalle bir LLM (Gemini, GPT vb.) tarafından üretilmiştir.")
                else:
                    st.success(f"Sonuç: **İNSAN** (%{human_score*100:.1f})")
                    st.info("Bu metin doğal insan dili özellikleri taşıyor.")
