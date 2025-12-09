# spontane
gelisine.py ilk sürüm 5 temel analiz.
gelisine-1.py son sürüm ¢▓▒░ son 460 gün baz alınmıştır.

Proje Özeti

Amaç: CSV formatında günlük hisse verilerini (Open, High, Low, Close, Volume) tarayıp gelişmiş sinyaller üreten bir analiz hattı. Sonuçları Excel ve PNG grafiklerine kaydeder; her hisse için bir dizi skor (0–10) ve kısa etiketler döner.
Konum: Ana betik: gelisine.py
Girdi dizini (varsayılan): metastock-gun-csv
Çıktılar: Excel raporu analiz_sonuclari_*.xlsx ve grafikleri plots/ klasörüne kaydeder.
Eklenen Temel Skorlar ve Anlamları

Akümülasyon (0–10): Uzun vadeli biriktirme eğilimi.
Dağıtım (0–10): Boşaltma / realize eğilimi.
Pump-Dump (0–10): Ani yön değişimleri + yüksek hacimli günler.
Fake Breakout (0–10): Breakout sonrası ters hareketlere işaret eder.
Momentum (0–10) + Trend Yönü: Fiyat trendine uyum.
Algoritmik Akım (0–10) + Aktivite Türü: Bar ritmi, fraktal yoğunluğu, micro-burst döngüleri → HFT/BOT/MIX/HMN.
Arz Emilimi (Supply-Absorption) (0–10) + Emiş Gücü: Uzun üst gölge + küçük gövde, hacim artışı + fiyat stabil, dar aralık/hacim kümelenmesi → kurumsal emiş sinyali (emoji ile gösterim).
Gizli Likidite / Iceberg (0–10) + Iceberg Türü: Dar ATR + yönlü kapanışlar + düşük hacim dönüşleri → Strong/Moderate/Weak ve Buy/Sell/Neutral tipleri.
Önemli Fonksiyonlar (kullanıcı için)

analyze_stock_advanced(df, lookback_days=None) — tek hisse gelişmiş analiz (dönen: skor sözlüğü).
analyze_all_stocks_advanced(folder_path, lookback_days=None, min_days=50) — klasördeki tüm CSV'leri analiz edip Excel+grafik üretir.
Skor fonksiyonları: calculate_algorithmic_footprint_score, calculate_supply_absorption_score, calculate_hidden_liquidity_score, calculate_momentum_score, vb.
Çalıştırma (Hızlı Başlangıç)

Gerekli paketler:
pandas, numpy, matplotlib, openpyxl
Örnek kurulum:
pip install pandas numpy matplotlib openpyxl
Programı çalıştırma (varsayılan dizinle):
python gelisine.py
Windows PowerShell'de UTF-8 çıktı gerekiyorsa:
$env:PYTHONIOENCODING='utf-8'; python gelisine.py
Konfigürasyon / Parametreler

lookback_days — None tüm veri; int = son N gün. (Ana blokta lookback_days=460 kullanılıyor.)
min_days — analiz için minimum satır sayısı (varsayılan 50).
CSV formatı: sütun isimleri Date, Open, High, Low, Close, Volume (başlık boşluk/harf farklılıkları normalize edilir).
Nasıl Yorumlamalı / Öneriler

Algoritmik skor yüksekse (ör. ≥7): HFT/BOT etkinliği güçlüdür — kısa süreli dalgalar, yüksek frekanslı hareketler beklenebilir.
Arz Emilimi yüksek + Emis Gücü 🟠/🔴: “Smart money” alımı olma ihtimali; sinyal kaçınılmaz değil — backtest önerilir.
Gizli Likidite (Iceberg) Strong-Buy/Strong-Sell: fiyatın görünürde hacim artışı olmadan destek/dirençten dönüyor olmasına işaret eder; dikkatle takip edin.
Hızlı Sonraki Adımlar (öneriyorum)

Backtest: Arz Emilimi ve Gizli Likidite sinyallerinin sonrası 5/10/20 günlük getirilerini hesapla.
Eşik Kalibrasyonu: eşiği veri ile (grid search) optimize et.
Görselleştirme: her hisse için emiş/iceberg günlerini işaretleyen küçük zaman serisi PNG’leri ekle.
Uyarılar: Arz Emilimi > X veya Gizli Likidite >= Y için otomatik filtre/CSV üret.
Kısa Notlar / Bilinen Durumlar

Konsolda Türkçe emoji/özel karakterlerle ilgili encoding problemleri görülebilir — PowerShell için PYTHONIOENCODING='utf-8' ayarlaması önerilir.
Fonksiyonlar veri kalitesine hassastır; eksik sütun veya boş/bozuk tarihler analizleri bozabilir.
