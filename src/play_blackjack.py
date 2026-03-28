import os
import argparse
import joblib
import pandas as pd
import numpy as np
import random
import time

# --- Model Yönetimi ---

def load_model_and_scaler(model_type, model_dir):
    """Belirtilen model tipine göre model ve scaler'ı yükler."""
    model_files = {
        'lr': ('logistic_regression_model.pkl', 'lr_scaler.pkl'),
        'nn': ('nn_model.pkl', 'nn_scaler.pkl'),
        'rf': ('rf_model.pkl', 'rf_scaler.pkl'),
        'xgb': ('xgb_model.pkl', 'xgb_scaler.pkl')
    }
    
    if model_type not in model_files:
        raise ValueError(f"Geçersiz model tipi: {model_type}. Seçenekler: lr, nn, rf, xgb")
    
    m_file, s_file = model_files[model_type]
    model_path = os.path.join(model_dir, m_file)
    scaler_path = os.path.join(model_dir, s_file)
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model veya Scaler bulunamadı: {model_path}")
    
    print(f"[*] {model_type.upper()} modeli ve scaler yükleniyor...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def get_model_recommendation(model, scaler, dealer_card, player_sum, is_soft):
    """Modeli kullanarak Hit veya Stand kararı verir."""
    feature_names = ['dealcard1', 'ply2cardsum', 'player_action', 'is_soft_hand']
    
    # Stand (0) senaryosu
    data_stand = pd.DataFrame([[dealer_card, player_sum, 0, int(is_soft)]], columns=feature_names)
    prob_stand = model.predict_proba(scaler.transform(data_stand))[0][1]
    
    # Hit (1) senaryosu
    data_hit = pd.DataFrame([[dealer_card, player_sum, 1, int(is_soft)]], columns=feature_names)
    prob_hit = model.predict_proba(scaler.transform(data_hit))[0][1]
    
    rec = "HIT" if prob_hit > prob_stand else "STAND"
    return rec, prob_hit, prob_stand

# --- Oyun Mantığı ---

class Deck:
    def __init__(self, num_decks=4):
        single_deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4
        self.cards = single_deck * num_decks
        random.shuffle(self.cards)
    
    def draw(self):
        if len(self.cards) < 20: 
            self.__init__()
        return self.cards.pop()

def calculate_hand(hand):
    """Elin toplamını ve soft hand olup olmadığını hesaplar."""
    total = sum(hand)
    aces = hand.count(11)
    is_soft = False
    
    if aces > 0:
        is_soft = True
    
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
        if aces == 0:
            is_soft = False
            
    return total, is_soft

def run_simulation(model, scaler, model_name, auto=False, rounds=1):
    deck = Deck(num_decks=4)
    stats = {"wins": 0, "losses": 0, "pushes": 0, "total": 0}
    
    print("\n" + "="*60)
    print(f" BLACKJACK SİMÜLASYONU BAŞLIYOR (Mod: {'OTOMATİK' if auto else 'İNTERAKTİF'}, Model: {model_name}) ")
    if auto:
        print(f" Hedef: {rounds} El Oynatılacak.")
    print("="*60)

    # El sayısı belirleme
    total_rounds = rounds if auto else 10**9 # Manuel modda pratik olarak sonsuz el
    
    for i in range(1, total_rounds + 1):
        if not auto:
            print(f"\n" + "-"*20)
            print(f"[ El #{i} Başlıyor ]")
            print("-"*20)
        elif i % 100 == 0:
            print(f"[*] Simülasyon ilerliyor: {i}/{rounds} el tamamlandı...")

        dealer_hand = [deck.draw()]
        player_hand = [deck.draw(), deck.draw()]
        
        # Oyuncu Turu
        while True:
            p_total, p_soft = calculate_hand(player_hand)
            d_card = dealer_hand[0]
            
            if p_total > 21:
                break
            if p_total == 21:
                break

            # Model Tavsiyesi
            rec, p_hit, p_stand = get_model_recommendation(model, scaler, d_card, p_total, p_soft)
            
            if not auto:
                print(f"Dealer: {d_card} | Eliniz: {player_hand} (Toplam: {p_total}, Soft: {p_soft})")
                print(f">>> TAVSİYE: {rec} (W_Hit: %{p_hit*100:.1f}, W_Stand: %{p_stand*100:.1f})")
                action = input("Aksiyon [H, S, Q]: ").lower()
                if action == 'q': return
            else:
                action = 'h' if rec == "HIT" else 's'

            if action == 'h':
                player_hand.append(deck.draw())
                continue
            else:
                break

        p_total, _ = calculate_hand(player_hand)
        
        # Dealer Turu & Sonuç
        if p_total > 21:
            stats["losses"] += 1
        else:
            # Dealer kart çeker
            while True:
                d_total, _ = calculate_hand(dealer_hand)
                if d_total < 17:
                    dealer_hand.append(deck.draw())
                else:
                    break
            
            d_total, _ = calculate_hand(dealer_hand)
            
            if d_total > 21:
                stats["wins"] += 1
                result = "KAZANDINIZ"
            elif p_total > d_total:
                stats["wins"] += 1
                result = "KAZANDINIZ"
            elif p_total < d_total:
                stats["losses"] += 1
                result = "KAYBETTİNİZ"
            else:
                stats["pushes"] += 1
                result = "BERABERE"
            
            if not auto:
                print(f"Sonuç: {result} (Siz: {p_total}, Dealer: {d_total})")

        stats["total"] += 1
        
        if not auto:
            cont = input("\nSonraki El için Enter, Çıkmak için Q: ").lower()
            if cont == 'q': break

    # Final Raporu
    win_rate = (stats["wins"] / stats["total"]) * 100 if stats["total"] > 0 else 0
    print("\n" + "="*40)
    print(f" SİMÜLASYON RAPORU ({model_name}) ")
    print("="*40)
    print(f"Toplam El Sayısı:  {stats['total']}")
    print(f"Galibiyet (Win):   {stats['wins']}")
    print(f"Mağlubiyet (Loss): {stats['losses']}")
    print(f"Beraberlik (Push): {stats['pushes']}")
    print("-" * 20)
    print(f"Kazanma Oranı:     %{win_rate:.2f}")
    print("="*40 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Blackjack Model Simülatörü")
    parser.add_argument("--model", type=str, default="xgb", choices=["lr", "nn", "rf", "xgb"], 
                        help="Kullanılacak model tipi (lr, nn, rf, xgb)")
    parser.add_argument("--auto", action="store_true", help="Otomatik simülasyon modunu açar.")
    parser.add_argument("--rounds", type=int, default=1, help="Simülasyon tur sayısı.")
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(project_root, 'models', 'saved_models')
    
    try:
        model, scaler = load_model_and_scaler(args.model, model_dir)
        run_simulation(model, scaler, args.model.upper(), auto=args.auto, rounds=args.rounds)
    except Exception as e:
        print(f"[!] Hata: {e}")

if __name__ == "__main__":
    main()
