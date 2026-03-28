import os
import pandas as pd
import numpy as np

def clean_data(input_file: str, output_file: str):
    print(f"[*] Veri dosyası yükleniyor: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"[!] HATA: {input_file} dosyası bulunamadı.")
        return
    
    initial_rows = df.shape[0]
    initial_cols = df.shape[1]
    
    print(f"[*] Başlangıç Boyutları: {initial_rows:,} Satır, {initial_cols} Kolon")

    # 1. Hatalı Krupiye Oyunlarının Silinmesi (5 kart çekip 17 altında kalan eller)
    if 'dealcard5' in df.columns and 'sumofdeal' in df.columns:
        df = df[~((df['dealcard5'] != 0) & (df['sumofdeal'] < 17))]
        print(f"[*] {initial_rows - df.shape[0]:,} Hatalı krupiye satırı tespit edildi ve silindi.")

    # Yeni Kolon: Aksiyon Tespiti (Hit=1, Stand=0)
    if 'card3' in df.columns and 'card4' in df.columns and 'card5' in df.columns:
        df['player_action'] = np.where((df['card3'] > 0) | (df['card4'] > 0) | (df['card5'] > 0), 1, 0)
        print(f"[*] 'player_action' başarıyla üretildi. (1=Hit, 0=Stand)")

    cols_to_drop = [
        'PlayerNo', 
        'dealcard2', 'dealcard3', 'dealcard4', 'dealcard5', 'sumofdeal', 
        'plybustbeat', 'dlbustbeat', 'plwinamt', 'dlwinamt',
        'sumofcards', 'card3', 'card4', 'card5', 'blkjck'
    ]
    
    unnamed_cols = [c for c in df.columns if "Unnamed" in c]
    cols_to_drop.extend(unnamed_cols)
    
    cols_to_drop_existing = [c for c in cols_to_drop if c in df.columns]
    
    df.drop(columns=cols_to_drop_existing, inplace=True)
    print(f"[-] {len(cols_to_drop_existing)} Kolon silindi: {cols_to_drop_existing}")

    # 2. As (Ace) Sızıntısının Düzeltilmesi! (1 -> 11 Dönüşümü)
    # Hem krupiyenin açık ilk kartında (dealcard1) hem de oyuncunun ilk 2 kartında (card1, card2) 
    # başlangıç aşamasında henüz karar anında 1 (bust korumalı as) kavramı yoktur. 
    # Oyun başvurduğunda hepsi sadece "As" is ve 11 kabul edilir.
    
    replace_cols = ['dealcard1', 'card1', 'card2']
    
    for col in replace_cols:
        if col in df.columns:
            # 1 olanları tespit edip 11'e güncelliyoruz.
            count_ones = (df[col] == 1).sum()
            df[col] = df[col].replace(1, 11)
            print(f"[*] {col} kolonundaki {count_ones:,} adet 1 değeri, 11 olarak (As) düzeltildi.")

    # 3. Eksik Verilerin Silinmesi (Dropna)
    df.dropna(inplace=True)
    final_rows = df.shape[0]
    final_cols = df.shape[1]
    
    dropped_nan = initial_rows - final_rows
    if dropped_nan > 0:
        print(f"[-] {dropped_nan:,} adet boş (NaN) satır silindi.")
    else:
        print("[*] Verilerde boş (NaN) satır tespit edilmedi.")

    print(f"[*] Final Boyutları: {final_rows:,} Satır, {final_cols} Kolon")

    # Çıktı klasörünü oluştur ve veriyi kaydet
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"[+] Temiz veri başarıyla kaydedildi: {output_file}")


if __name__ == "__main__":
    
    # Proje kök dizinini ayarlıyoruz (Dosya src/data_process/clean_data.py içerisinde)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    raw_file = os.path.join(project_root, 'data', 'raw', 'blkjckhands.csv')
    clean_file = os.path.join(project_root, 'data', 'processed', 'clean_blkjckhands.csv')
    
    clean_data(raw_file, clean_file)
