import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_random_forest(data_path: str, model_save_path: str, scaler_save_path: str):
    print(f"[*] Veri yükleniyor: {data_path}")
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    print(f"[*] Özellikler (Features): {list(X.columns)}")
    print(f"[*] Toplam Veri: {len(df):,} satır.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("[*] Veri Eğitim ve Test (80/20) olarak bölündü.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[*] Modeller arası stabilite için veriler ölçeklendirildi (StandardScaler).")
    
    print("[-] Random Forest (Rastgele Orman) modeli oluşturuluyor...")
    # Parametreler (işlemciyi iyi kullanması için n_jobs=-1, daha hızlı eğitim)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15, # Aşırı öğrenmeyi (overfitting) engellemek için sınır
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("[-] Random Forest modeli eğitiliyor... Lütfen bekleyin.")
    model.fit(X_train_scaled, y_train)

    print("[-] Test verisi üzerinden tahminler yapılıyor...")
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"[+] Random Forest Eğitimi Tamamlandı!")
    print(f"[+] Test Doğruluk Oranı (Accuracy): %{acc * 100:.2f}")
    
    print("\n--- Sınıflandırma Raporu (Classification Report) ---")
    print(classification_report(y_test, y_pred, target_names=["Kaybedilen/Push (0)", "Kazanılan (1)"]))
    
    print("--- Karmaşıklık Matrisi (Confusion Matrix) ---")
    print(confusion_matrix(y_test, y_pred))
    print("="*50 + "\n")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    joblib.dump(model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print(f"[+] Model başarıyla kaydedildi: {model_save_path}")
    print(f"[+] Scaler başarıyla kaydedildi: {scaler_save_path}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_file = os.path.join(project_root, 'data', 'processed', 'ml_ready_blkjckhands.csv')
    
    model_dir = os.path.join(project_root, 'models', 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_file = os.path.join(model_dir, 'rf_model.pkl')
    scaler_file = os.path.join(model_dir, 'rf_scaler.pkl')
    
    train_random_forest(data_file, model_file, scaler_file)
