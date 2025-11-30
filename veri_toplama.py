import arxiv
import pandas as pd
import time

def fetch_arxiv_data(max_results=3000):
    print(f"Veri çekme işlemi başladı. Hedef: {max_results} makale özeti...")
    
    # Arxiv istemcisini oluştur
    client = arxiv.Client()
    
    # Arama kriterleri: Yapay Zeka (cs.AI), Makine Öğrenmesi (cs.LG) veya Dil İşleme (cs.CL)
    search = arxiv.Search(
        query = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL",
        max_results = max_results,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )

    data = []
    count = 0

    # Sonuçları gez
    try:
        results = client.results(search)
        for r in results:
            # Satır sonlarını boşlukla değiştirip metni tek satır haline getirelim
            cleaned_text = r.summary.replace("\n", " ").strip()
            
            data.append({
                "text": cleaned_text,
                "label": "human"  # Etiketimiz: İnsan
            })
            
            count += 1
            if count % 100 == 0:
                print(f"{count} adet veri çekildi...")
                
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

    # DataFrame'e çevir ve kaydet
    df = pd.DataFrame(data)
    
    # Dosya ismini proje isterlerine uygun verelim
    filename = "human_data.csv"
    df.to_csv(filename, index=False)
    
    print(f"\nİşlem tamamlandı! Toplam {len(df)} veri '{filename}' dosyasına kaydedildi.")
    print("İlk 5 veri örneği:")
    print(df.head())

if __name__ == "__main__":
    fetch_arxiv_data()