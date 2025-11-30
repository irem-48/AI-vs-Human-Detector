import google.generativeai as genai

# ANAHTARINI BURAYA YAPIŞTIR
API_KEY = "AIzaSyAnGj3Op3Rz2CItGEEPG3fsHfxNwM7E30g" 

genai.configure(api_key=API_KEY)

print("🔍 Erişilebilir Modeller Listeleniyor...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Model: {m.name}")
except Exception as e:
    print(f"❌ Hata: {e}")