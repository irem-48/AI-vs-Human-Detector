import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_models():
    print("🚀 Model eğitimi başlıyor...")

    # 1. Verileri Yükle
    try:
        # Hata durumlarına karşı on_bad_lines='skip' ekledik
        df_human = pd.read_csv("human_data.csv", on_bad_lines='skip')
        df_ai = pd.read_csv("ai_data.csv", on_bad_lines='skip')
    except FileNotFoundError:
        print("Hata: CSV dosyaları bulunamadı!")
        return

    # Verileri birleştir
    df = pd.concat([df_human, df_ai], ignore_index=True)
    
    # Temizlik (Boş verileri ve tekrar edenleri sil)
    print(f"Temizlik öncesi veri sayısı: {len(df)}")
    df.dropna(subset=['text', 'label'], inplace=True)
    df.drop_duplicates(subset=['text'], inplace=True)
    print(f"Temizlik sonrası net veri sayısı: {len(df)}")

    # 2. Metni Sayısallaştır (TF-IDF)
    print("Metinler vektörleştiriliyor (TF-IDF)...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['text'].astype(str)) # String olduğundan emin olalım
    y = df['label']

    # 3. Eğitim ve Test Olarak Ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Modelleri Tanımla
    models = {
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=50), # Hız için 50 ağaç
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ""

    # 5. Modelleri Eğit
    print("\n📊 SONUÇLAR:")
    print("-" * 30)
    
    for name, model in models.items():
        print(f"⏳ {name} eğitiliyor...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"✅ {name} Başarısı: %{acc * 100:.2f}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name

    print("-" * 30)
    print(f"🏆 KAZANAN MODEL: {best_model_name} (%{best_accuracy * 100:.2f})")

    # 6. Kaydet
    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("💾 best_model.pkl ve vectorizer.pkl kaydedildi.")

if __name__ == "__main__":
    train_models()