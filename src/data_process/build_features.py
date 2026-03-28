import os
import pandas as pd
import numpy as np

def build_features(input_file: str, output_file: str):
    print(f"[*] Özellik (Feature) üretimi başlatılıyor: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"[!] HATA: {input_file} dosyası bulunamadı.")
        return

    # 1. Soft Hand (Yumuşak El) Tespiti
    # İlk 2 karttan herhangi birisi 11 (As) ise, bu "Soft Hand" (Yumuşak El) olarak kabul edilir.
    df['is_soft_hand'] = np.where((df['card1'] == 11) | (df['card2'] == 11), 1, 0)
    
    count_soft = df['is_soft_hand'].sum()
    print(f"[+] 'is_soft_hand' kolonu eklendi. Toplam 'Soft Hand' (As içeren) oyun sayısı: {count_soft:,}")

    # 2. Makine Öğrenmesi İçin Hedef Değişkeni (Target) Düzenleme
    # 'winloss' kolonunu ML modellerinin anlayacağı sayısal bir 'is_win' kolonuna çeviriyoruz.
    # Win = 1, Loss = 0, Push = 0 (Beraberlikleri kazanmamış sayıyoruz veya modelinize göre ayırabilirsiniz)
    df['target'] = np.where(df['winloss'] == 'Win', 1, 0)
    
    count_wins = df['target'].sum()
    print(f"[+] 'target' kolonu eklendi (ML Sınıflandırma İçin). Toplam Kazanan Oyun Sayısı: {count_wins:,}")
    
    # 3. Özellik Seçimi (Sadece Modele Girecekleri Bırakma)
    df.drop(columns=['winloss', 'card1', 'card2'], inplace=True, errors='ignore')
    print("[-] 'winloss', 'card1' ve 'card2' kolonları model eğitimi için düşürüldü.")

    # Çıktı klasörünü oluştur ve veriyi kaydet
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"[+] ML'e hazır veri başarıyla kaydedildi: {output_file}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    clean_file = os.path.join(project_root, 'data', 'processed', 'clean_blkjckhands.csv')
    ml_ready_file = os.path.join(project_root, 'data', 'processed', 'ml_ready_blkjckhands.csv')
    
    build_features(clean_file, ml_ready_file)
