import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== SKOR HESAPLAMA FONKSİYONLARI ====================
# (Fonksiyonlar aynı, sadece optimize edilmiş)

def calculate_accumulation_score(df):
    """Akümülasyon skorunu hesaplar (0-10)"""
    if len(df) < 50:
        return 0
    
    temp = df.copy()
    period = min(20, len(df) // 3)  # Dinamik period
    
    # Hesaplamalar...
    temp['range'] = temp['High'] - temp['Low']
    temp['range'] = temp['range'].replace(0, 0.001)
    temp['price_position'] = (temp['Close'] - temp['Low']) / temp['range']
    
    temp['volume_norm'] = temp['Volume'] / temp['Volume'].rolling(period).mean()
    
    # Akümülasyon sinyali
    mask_middle = (temp['price_position'] >= 0.3) & (temp['price_position'] <= 0.7)
    temp['accum_signal'] = 0
    temp.loc[mask_middle, 'accum_signal'] = temp['volume_norm'] * 0.7
    
    # Trend bonus
    sma_period = min(50, len(df) // 2)
    sma = temp['Close'].rolling(sma_period).mean()
    temp['trend_bonus'] = np.where(temp['Close'] > sma, 0.3, 0)
    
    # Son dönem skoru
    lookback = min(period, len(temp))
    recent_data = temp.iloc[-lookback:]
    
    if len(recent_data) > 0:
        accum_score = (recent_data['accum_signal'] + recent_data['trend_bonus']).mean()
        score = min(10, max(0, accum_score * 5))
        return round(score, 2)
    return 0

def calculate_distribution_score(df):
    """Dağıtım skorunu hesaplar (0-10)"""
    if len(df) < 50:
        return 0
    
    temp = df.copy()
    period = min(20, len(df) // 3)
    
    # Hesaplamalar...
    temp['range'] = temp['High'] - temp['Low']
    temp['range'] = temp['range'].replace(0, 0.001)
    temp['price_position'] = (temp['Close'] - temp['Low']) / temp['range']
    
    temp['close_change'] = temp['Close'].pct_change()
    temp['is_negative_close'] = (temp['Close'] < temp['Open']).astype(int)
    
    temp['volume_norm'] = temp['Volume'] / temp['Volume'].rolling(period).mean()
    
    # Dağıtım sinyali
    mask_upper = temp['price_position'] > 0.7
    mask_negative = temp['is_negative_close'] == 1
    temp['dist_signal'] = 0
    temp.loc[mask_upper & mask_negative, 'dist_signal'] = temp['volume_norm'] * 0.8
    
    # Düşüş trendi
    sma_period = min(20, len(df) // 3)
    sma = temp['Close'].rolling(sma_period).mean()
    temp['down_trend_bonus'] = np.where(temp['Close'] < sma, 0.4, 0)
    
    # Skor
    lookback = min(period, len(temp))
    recent_data = temp.iloc[-lookback:]
    
    if len(recent_data) > 0:
        dist_score = (recent_data['dist_signal'] + recent_data['down_trend_bonus']).mean()
        score = min(10, max(0, dist_score * 4))
        return round(score, 2)
    return 0

def calculate_pumpdump_score(df):
    """Pump-Dump skorunu hesaplar (0-10)"""
    if len(df) < 30:
        return 0
    
    temp = df.copy()
    period = min(10, len(df) // 4)
    
    # Hacim spike
    vol_avg = temp['Volume'].rolling(period).mean()
    temp['volume_spike'] = temp['Volume'] / vol_avg
    temp['high_vol_spike'] = (temp['volume_spike'] > 2.0).astype(int)
    
    # Volatilite
    temp['returns'] = temp['Close'].pct_change()
    temp['abs_returns'] = abs(temp['returns'])
    
    # Gap
    temp['gap'] = abs(temp['Open'] - temp['Close'].shift(1)) / temp['Close'].shift(1).replace(0, 0.001)
    temp['high_gap'] = (temp['gap'] > 0.03).astype(int)
    
    # Range
    temp['daily_range'] = (temp['High'] - temp['Low']) / temp['Low'].replace(0, 0.001)
    temp['wide_range'] = (temp['daily_range'] > 0.05).astype(int)
    
    # PD skoru
    temp['pd_score_raw'] = (temp['high_vol_spike'] * 3 + 
                           temp['high_gap'] * 2 + 
                           temp['wide_range'] * 2 + 
                           (temp['abs_returns'] > 0.04).astype(int) * 3)
    
    # Son period
    lookback = min(period * 2, len(temp))
    recent_pd = temp['pd_score_raw'].iloc[-lookback:].mean()
    
    score = min(10, recent_pd)
    return round(score, 2)

def calculate_fakebreakout_score(df):
    """Fake Breakout skorunu hesaplar (0-10)"""
    if len(df) < 60:
        return 0
    
    temp = df.copy()
    period = min(20, len(df) // 3)
    
    # Destek/direnç
    lookback_period = min(20, len(df) // 3)
    temp['high_20'] = temp['High'].rolling(lookback_period).max()
    temp['low_20'] = temp['Low'].rolling(lookback_period).min()
    
    # Breakout tespiti
    resistance_break = (temp['Close'] > temp['high_20'].shift(1)).astype(int)
    support_break = (temp['Close'] < temp['low_20'].shift(1)).astype(int)
    
    # Fake breakout kontrolü
    temp['fake_up'] = 0
    temp['fake_down'] = 0
    
    for i in range(2, len(temp)-1):
        if resistance_break.iloc[i-1] == 1:
            # Sonraki 2 gün %2'den fazla düşüş
            min_close = min(temp['Close'].iloc[i], temp['Close'].iloc[i+1])
            if min_close < temp['Close'].iloc[i-1] * 0.98:
                temp['fake_up'].iloc[i] = 1
        
        if support_break.iloc[i-1] == 1:
            # Sonraki 2 gün %2'den fazla yükseliş
            max_close = max(temp['Close'].iloc[i], temp['Close'].iloc[i+1])
            if max_close > temp['Close'].iloc[i-1] * 1.02:
                temp['fake_down'].iloc[i] = 1
    
    # Hacim
    vol_avg = temp['Volume'].rolling(period).mean()
    temp['high_volume'] = (temp['Volume'] > vol_avg * 1.5).astype(int)
    
    temp['fake_signal'] = ((temp['fake_up'] + temp['fake_down']) * temp['high_volume'] * 2)
    
    # Fake oranı
    lookback = min(period * 2, len(temp))
    recent_data = temp.iloc[-lookback:]
    
    fake_count = recent_data['fake_signal'].sum()
    total_breaks = (recent_data['high_20'].notna() & recent_data['low_20'].notna()).sum()
    
    if total_breaks > 5:  # Yeterli breakout varsa
        fake_ratio = fake_count / total_breaks
        score = min(10, fake_ratio * 20)
        return round(score, 2)
    
    return 0

def calculate_momentum_score(df):
    """Momentum skorunu hesaplar (0-10)"""
    if len(df) < 50:
        return 0, "→"
    
    temp = df.copy()
    
    # SMA'lar
    sma_20 = temp['Close'].rolling(min(20, len(df)//3)).mean()
    sma_50 = temp['Close'].rolling(min(50, len(df)//2)).mean()
    
    # Trend yönü
    ma_alignment = 0
    if sma_20.iloc[-1] > sma_50.iloc[-1]:
        ma_alignment = 1  # ↑
    elif sma_20.iloc[-1] < sma_50.iloc[-1]:
        ma_alignment = -1  # ↓
    
    # RSI
    delta = temp['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=min(14, len(df)//4)).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=min(14, len(df)//4)).mean()
    rs = gain / loss.replace(0, 0.001)
    rsi = 100 - (100 / (1 + rs))
    
    rsi_signal = 0
    if rsi.iloc[-1] > 60:
        rsi_signal = 0.5
    elif rsi.iloc[-1] < 40:
        rsi_signal = -0.5
    
    # Trend gücü (basit)
    lookback = min(20, len(df)//3)
    if lookback > 0:
        price_change = (temp['Close'].iloc[-1] - temp['Close'].iloc[-lookback]) / temp['Close'].iloc[-lookback]
        trend_strength = min(1, max(-1, price_change * 10))  # Normalize
    else:
        trend_strength = 0
    
    # Nihai momentum
    momentum_raw = (ma_alignment * 3 + rsi_signal * 2 + trend_strength * 3)
    
    # 0-10'a çevir
    if momentum_raw > 0:
        score = 5 + (momentum_raw * 1.5)
    else:
        score = 5 + (momentum_raw * 1.5)
    
    score = min(10, max(0, score))
    
    # Yön belirle
    if momentum_raw > 0.5:
        direction = "↑"
    elif momentum_raw < -0.5:
        direction = "↓"
    else:
        direction = "→"
    
    return round(score, 2), direction

# ==================== GELİŞMİŞ ANALİZ FONKSİYONU ====================
def analyze_stock_advanced(df, lookback_days=None):
    """
    Gelişmiş analiz fonksiyonu - Kullanıcı belirler veya tüm veriyi kullanır
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Hisse verisi (Date index, Open, High, Low, Close, Volume)
    lookback_days : int or None
        None: Tüm veriyi kullan
        int: Son N günü kullan
    """
    if df.empty or len(df) < 30:
        return None
    
    # Kullanılacak veriyi seç
    if lookback_days is None:
        analysis_df = df.copy()
        period_info = f"Tüm veri ({len(df)} gün)"
    else:
        lookback_days = min(lookback_days, len(df))
        analysis_df = df.iloc[-lookback_days:].copy()
        period_info = f"Son {lookback_days} gün"
    
    print(f"\n📊 Analiz periyodu: {period_info}")
    
    # Skorları hesapla
    scores = {
        'Akümülasyon': calculate_accumulation_score(analysis_df),
        'Dağıtım': calculate_distribution_score(analysis_df),
        'Pump-Dump': calculate_pumpdump_score(analysis_df),
        'Fake Breakout': calculate_fakebreakout_score(analysis_df),
    }
    
    mom_score, mom_dir = calculate_momentum_score(analysis_df)
    scores['Momentum'] = mom_score
    scores['Trend Yönü'] = mom_dir
    
    # Ek bilgiler
    scores['İlk Tarih'] = df.index[0].strftime('%d.%m.%Y')
    scores['Son Tarih'] = df.index[-1].strftime('%d.%m.%Y')
    scores['Toplam Gün'] = len(df)
    scores['Analiz Günü'] = len(analysis_df)
    
    # Fiyat bilgileri
    scores['Son Fiyat'] = round(df['Close'].iloc[-1], 2)
    scores['Değişim 1G (%)'] = round(((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100, 2) if len(df) > 1 else 0
    
    # Ortalama hacim
    avg_volume = df['Volume'].mean()
    scores['Ort. Hacim'] = f"{avg_volume:,.0f}"
    
    return scores

# ==================== TÜM DİZİNİ ANALİZ ET ====================
def analyze_all_stocks_advanced(folder_path, lookback_days=None, min_days=50):
    """
    Tüm CSV dosyalarını analiz eder
    
    Parameters:
    -----------
    folder_path : str
        CSV dosyalarının bulunduğu dizin
    lookback_days : int or None
        None: Tüm veriyi kullan
        int: Son N günü kullan
    min_days : int
        Minimum gün sayısı (daha az olanlar analiz edilmez)
    """
    folder = Path(folder_path)
    csv_files = list(folder.glob("*.csv"))
    
    if not csv_files:
        print(f"❌ {folder_path} dizininde CSV dosyası bulunamadı!")
        return
    
    print(f"🔍 {len(csv_files)} CSV dosyası bulundu. Analiz başlıyor...")
    print("=" * 120)
    
    all_results = []
    
    for i, csv_file in enumerate(csv_files, 1):
        try:
            # CSV'yi oku
            df = pd.read_csv(csv_file)
            
            # Sütun isimlerini standartlaştır
            df.columns = [col.strip().title() for col in df.columns]
            
            # Date sütununu işle
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date'])
                df = df.set_index('Date').sort_index()
            
            # Gerekli sütunları kontrol et
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"⚠️  {csv_file.stem:25} | Eksik sütunlar: {missing_cols}")
                continue
            
            # Veriyi temizle
            df = df[required_cols].dropna()
            
            if len(df) < min_days:
                print(f"⚠️  {csv_file.stem:25} | Yetersiz veri: {len(df)} gün (min {min_days})")
                continue
            
            # Analiz yap
            scores = analyze_stock_advanced(df, lookback_days)
            
            if scores:
                result = {
                    'Hisse': csv_file.stem,
                    'Kod': csv_file.stem.split('_')[0] if '_' in csv_file.stem else csv_file.stem,
                    **scores
                }
                all_results.append(result)

                # Konsola yazdır
                print(f"✅ {i:3d}. {csv_file.stem:25} | "
                      f"A:{scores['Akümülasyon']:4.1f} "
                      f"D:{scores['Dağıtım']:4.1f} "
                      f"P:{scores['Pump-Dump']:4.1f} "
                      f"F:{scores['Fake Breakout']:4.1f} "
                      f"M:{scores['Momentum']:4.1f}{scores['Trend Yönü']} "
                      f"| F:{scores['Son Fiyat']:8.2f} "
                      f"Δ:{scores['Değişim 1G (%)']:+6.2f}%")
            
        except Exception as e:
            print(f"❌ {i:3d}. {csv_file.stem:25} | HATA: {str(e)[:50]}...")
            continue
    
    # Sonuçları DataFrame'e çevir
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Sütun sıralaması
        column_order = ['Hisse', 'Kod', 'Son Fiyat', 'Değişim 1G (%)', 'Akümülasyon', 
                       'Dağıtım', 'Pump-Dump', 'Fake Breakout', 'Momentum', 'Trend Yönü',
                       'Ort. Hacim', 'İlk Tarih', 'Son Tarih', 'Toplam Gün', 'Analiz Günü']
        
        # Eksik sütunları filtrele
        column_order = [col for col in column_order if col in results_df.columns]
        results_df = results_df[column_order]
        
        # Sıralama (Momentuma göre)
        results_df = results_df.sort_values('Momentum', ascending=False)
        
        # Excel'e kaydet
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        if lookback_days:
            output_file = folder / f"analiz_sonuclari_son{lookback_days}gun_{timestamp}.xlsx"
        else:
            output_file = folder / f"analiz_sonuclari_tumveri_{timestamp}.xlsx"
        
        results_df.to_excel(output_file, index=False)
        
        # Konsola özet
        print("\n" + "=" * 120)
        print("📋 ANALİZ SONUÇLARI ÖZETİ")
        print("=" * 120)
        print(f"Toplam analiz edilen hisse: {len(results_df)}")
        print(f"Excel dosyası: {output_file.name}")
        
        # İstatistikler
        print("\n📊 SKOR ORTALAMALARI:")
        score_cols = ['Akümülasyon', 'Dağıtım', 'Pump-Dump', 'Fake Breakout', 'Momentum']
        for col in score_cols:
            if col in results_df.columns:
                avg = results_df[col].mean()
                print(f"  {col:15}: {avg:5.2f}")
        
        # En yüksek skorlu hisseler
        for col in score_cols:
            if col in results_df.columns:
                top = results_df.nlargest(10, col)[['Hisse', col]]
                print(f"\n🏆 EN YÜKSEK {col.upper()}:")
                for _, row in top.iterrows():
                    print(f"  {row['Hisse']:25}: {row[col]:5.1f}")
        
        return results_df, output_file
    else:
        print("\n❌ Hiçbir hisse başarıyla analiz edilemedi!")
        return None, None

# ==================== INTERAKTİF ÇALIŞTIRMA ====================
def interactive_analysis():
    """Kullanıcıdan girdi alarak interaktif analiz yapar"""
    print("🎯 HİSSE ANALİZ SİSTEMİ")
    print("=" * 50)
    
    # Dizin yolunu al
    default_path = r"D:\new\yeniden\metastock-gun-csv"
    folder_path = input(f"\n📁 CSV dosyalarının bulunduğu dizin [{default_path}]: ").strip()
    if not folder_path:
        folder_path = default_path
    
    # Periyodu seç
    print("\n📅 ANALİZ PERİYODU SEÇİMİ:")
    print("  1. Tüm veriyi kullan")
    print("  2. Son N günü kullan")
    
    choice = input("\nSeçiminiz (1 veya 2): ").strip()
    
    if choice == '2':
        while True:
            try:
                lookback_days = int(input("\nKaç günlük veri analiz edilsin? (örn: 30, 60, 90): "))
                if lookback_days >= 20:
                    break
                else:
                    print("⚠️  En az 20 gün giriniz!")
            except:
                print("⚠️  Geçerli bir sayı giriniz!")
    else:
        lookback_days = None
    
    # Minimum gün sayısı
    min_days = input(f"\n📊 Minimum gün sayısı (varsayılan: 50): ").strip()
    min_days = int(min_days) if min_days.isdigit() else 50
    
    print("\n" + "=" * 50)
    print("⏳ Analiz başlıyor...")
    
    # Analizi çalıştır
    results, output_file = analyze_all_stocks_advanced(
        folder_path=folder_path,
        lookback_days=lookback_days,
        min_days=min_days
    )
    
    if results is not None:
        print(f"\n✅ Analiz tamamlandı!")
        print(f"📁 Sonuçlar kaydedildi: {output_file}")
        
        # Ek işlemler
        print("\n🔧 EK İŞLEMLER:")
        print("  1. Belirli bir hisseyi detaylı analiz et")
        print("  2. Skorlara göre filtrele")
        print("  3. Çıkış")
        
        choice2 = input("\nSeçiminiz: ").strip()
        
        if choice2 == '1':
            hisse = input("\n📈 Hangi hisseyi detaylı analiz etmek istersiniz? (Hisse kodunu girin): ").strip().upper()
            # Burada detaylı analiz fonksiyonu eklenebilir
            print(f"\n⚠️  Detaylı analiz özelliği eklenecek...")
        
        elif choice2 == '2':
            print("\n🎯 FİLTRELEME SEÇENEKLERİ:")
            print("  1. Yüksek Akümülasyon (>7)")
            print("  2. Yüksek Dağıtım (>7)")
            print("  3. Düşük Pump-Dump (<3)")
            print("  4. Yüksek Momentum (>7)")
            
            filter_choice = input("\nFiltre seçiniz: ").strip()
            
            if filter_choice == '1':
                filtered = results[results['Akümülasyon'] > 7]
            elif filter_choice == '2':
                filtered = results[results['Dağıtım'] > 7]
            elif filter_choice == '3':
                filtered = results[results['Pump-Dump'] < 3]
            elif filter_choice == '4':
                filtered = results[results['Momentum'] > 7]
            else:
                filtered = results
            
            print(f"\n📋 Filtrelenmiş {len(filtered)} hisse:")
            print(filtered[['Hisse', 'Son Fiyat', 'Akümülasyon', 'Dağıtım', 'Pump-Dump', 'Momentum']].to_string(index=False))

# ==================== DOĞRUDAN ÇALIŞTIRMA ====================
if __name__ == "__main__":
    # Seçenek 1: Interaktif mod
    # interactive_analysis()
    
    # Seçenek 2: Direkt çalıştırma
    folder_path = r"D:\new\yeniden\metastock-gun-csv"
    
    # SEÇENEKLER:
    # 1. Tüm veriyi kullan:
    # results, output_file = analyze_all_stocks_advanced(folder_path, lookback_days=None)
    
    # 2. Son 60 günü kullan:
    results, output_file = analyze_all_stocks_advanced(folder_path, lookback_days=460)
    
    # 3. Son 90 günü kullan:
    # results, output_file = analyze_all_stocks_advanced(folder_path, lookback_days=90)


# ==================== GÖRSEL TOP10 ÇIKTISI ====================
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("⚠️  matplotlib yüklenmemiş. Grafik oluşturulamıyor.")
else:
    if results is not None and isinstance(results, pd.DataFrame):
        df_all = results.copy()
        
        score_cols = ['Akümülasyon', 'Dağıtım', 'Pump-Dump', 'Fake Breakout', 'Momentum']
        available = [c for c in score_cols if c in df_all.columns]
        
        if available:
            plots_dir = Path(folder_path) / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Kombine görsel (grid)
            n = len(available)
            cols = 2
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
            axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            
            for i, col in enumerate(available):
                ax = axes_flat[i]
                top10 = df_all.nlargest(10, col)[['Hisse', col]].dropna()
                if top10.empty:
                    ax.text(0.5, 0.5, 'Veri yok', ha='center', va='center', fontsize=12)
                    ax.set_title(f'{col}', fontsize=12, fontweight='bold')
                    ax.axis('off')
                    continue
                top10_sorted = top10.iloc[::-1]
                ax.barh(range(len(top10_sorted)), top10_sorted[col].values, color=f'C{i}', alpha=0.8)
                ax.set_yticks(range(len(top10_sorted)))
                ax.set_yticklabels(top10_sorted['Hisse'].values, fontsize=9)
                ax.set_title(f'Top 10 - {col}', fontsize=11, fontweight='bold')
                ax.set_xlabel(col, fontsize=10)
                ax.grid(axis='x', alpha=0.3)
            
            # Eksik eksenleri kaldır
            for j in range(i + 1, len(axes_flat)):
                try:
                    fig.delaxes(axes_flat[j])
                except Exception:
                    pass
            
            fig.tight_layout()
            combined_png = plots_dir / 'top10_all_scores.png'
            fig.savefig(combined_png, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Bireysel grafikler
            for i, col in enumerate(available):
                top10 = df_all.nlargest(10, col)[['Hisse', col]].dropna()
                if not top10.empty:
                    top10_sorted = top10.iloc[::-1]
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    ax2.barh(range(len(top10_sorted)), top10_sorted[col].values, color=f'C{i}', alpha=0.8)
                    ax2.set_yticks(range(len(top10_sorted)))
                    ax2.set_yticklabels(top10_sorted['Hisse'].values, fontsize=10)
                    ax2.set_title(f'Top 10 - {col}', fontsize=13, fontweight='bold')
                    ax2.set_xlabel(col, fontsize=11)
                    ax2.grid(axis='x', alpha=0.3)
                    fig2.tight_layout()
                    out_file = plots_dir / f"top10_{col.replace(' ', '_')}.png"
                    fig2.savefig(out_file, dpi=150, bbox_inches='tight')
                    plt.close(fig2)
            
            print(f"\n✓ Grafikler kaydedildi:")
            print(f"  - {combined_png}")
            print(f"  - {plots_dir / 'top10_*.png'}")
        else:
            print('⚠️  Skor sütunu bulunamadı.')
    else:
        print('⚠️  results boş veya None; analiz başarısız olmuş.')