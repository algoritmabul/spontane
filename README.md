# 📈 Gelişmiş Hisse Senedi Analiz Sistemi (Advanced Stock Analyzer)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

Finansal piyasalarda **Smart Money (Akıllı Para)**, **Algoritmik/HFT Aktiviteleri**, **Kurumsal İşlemler** ve **Piyasa Manipülasyonu** izlerini tespit eden gelişmiş bir hisse senedi analiz sistemidir.

Metastock/CSV formatındaki günlük OHLCV verilerini analiz ederek, her hisse için 0-10 arasında skore edilmiş 8 farklı piyasa davranışı ölçütü sunar.

---

## ✨ Özellikler

- 🔍 **Toplu Analiz:** Bir klasördeki tüm CSV dosyalarını otomatik analiz eder
- 🧠 **8 Farklı Analiz Modülü:** Akümülasyon, Dağıtım, Pump-Dump, Fake Breakout, Momentum, Algoritmik İz, Arz Emilimi, Gizli Likidite
- 📊 **Görsel Çıktılar:** Top 10 hisseler için otomatik grafik üretimi
- 💾 **Excel Rapor:** Tüm sonuçları tarih-saat damgalı Excel dosyası olarak kaydeder
- 🎯 **İnteraktif Mod:** Komut satırından yönlendirme ile kolay kullanım
- ⚡ **Dinamik Periyot:** Tüm veri veya son N gün seçeneği

---

## 🚀 Kurulum

### Gereksinimler

```bash
pip install pandas numpy matplotlib openpyxl
```

### Klonlama

```bash
git clone https://github.com/kullanici/hisse-analiz-sistemi.git
cd hisse-analiz-sistemi
```

---

## 📖 Kullanım

### 1. Veri Klasörünü Hazırlayın

Analiz etmek istediğiniz CSV dosyalarını tek bir klasörde toplayın. Dosyalar şu sütunları içermelidir:

| Sütun | Açıklama |
|-------|----------|
| `Date` | Tarih (YYYY-MM-DD) |
| `Open` | Açılış fiyatı |
| `High` | En yüksek fiyat |
| `Low` | En düşük fiyat |
| `Close` | Kapanış fiyatı |
| `Volume` | Hacim |

### 2. Doğrudan Çalıştırma

Kodun sonundaki `if __name__ == "__main__":` bloğunu düzenleyin:

```python
folder_path = r"C:\data\hisse_csv"  # Kendi yolunuzu yazın

# Tüm veriyi analiz et
results, output_file = analyze_all_stocks_advanced(folder_path, lookback_days=None)

# VEYA son 90 günü analiz et
results, output_file = analyze_all_stocks_advanced(folder_path, lookback_days=90)
```

```bash
python stock_analyzer.py
```

### 3. İnteraktif Mod

```python
if __name__ == "__main__":
    interactive_analysis()  # Bu satırı aktif edin
```

Program size:
- 📁 CSV klasörü yolu
- 📅 Analiz periyodu (tüm veri / son N gün)
- 📊 Minimum gün sayısı

sorularını soracaktır.

---

## 📊 Skorların Anlamı (En Önemli Bölüm)

Tüm skorlar **0 ile 10** arasında değer alır. Yüksek skor = güçlü sinyal.

### 🟢 1. Akümülasyon Skoru (Accumulation)
> Büyük oyuncuların (Smart Money) hisseyi **sessizce topladığı** dönemleri tespit eder.

| Skor | Anlam |
|------|-------|
| 0-3 | Akümülasyon yok / Zayıf |
| 4-6 | Orta seviye toplama |
| **7-10** | **Güçlü akümülasyon - Yükseliş öncesi sinyal** |

**Tespit yöntemi:** Fiyat dar bir aralıkta, orta seviyede hacimle tutunuyor + trend yukarı yönde

---

### 🔴 2. Dağıtım Skoru (Distribution)
> Büyük oyuncuların ellerindeki malı **küçük yatırımcıya satmaya başladığı** dönemler.

| Skor | Anlam |
|------|-------|
| 0-3 | Dağıtım yok |
| 4-6 | Orta seviye satış baskısı |
| **7-10** | **Güçlü dağıtım - Düşüş öncesi sinyal** |

**Tespit yöntemi:** Fiyat üst bölgelerde + negatif kapanışlar + ortalamanın üstünde hacim

---

### ⚠️ 3. Pump-Dump Skoru
> Şişirme ve bırakma (manipülasyon) aktivitelerini ölçer.

| Skor | Anlam |
|------|-------|
| 0-3 | Normal piyasa davranışı |
| 4-6 | Şüpheli volatilite |
| **7-10** | **Yüksek manipülasyon riski - Dikkat!** |

**Tespit yöntemi:** Hacim spike'ları + yüksek gap'ler + geniş günlük range + ani hareketler

---

### 🎭 4. Fake Breakout (Sahte Kırılım)
> Yatırımcıyı tuzağa düşüren sahte destek/direnç kırılımlarını yakalar.

| Skor | Anlam |
|------|-------|
| 0-3 | Güvenilir kırılımlar |
| 4-6 | Karışık sinyaller |
| **7-10** | **Çok sayıda sahte kırılım - Tuzak riski yüksek** |

**Tespit yöntemi:** 20 günlük zirve/dip kırıldıktan sonraki 2 gün içinde %2+ ters hareket + yüksek hacim

---

### 🚀 5. Momentum Skoru ve Trend Yönü
> Mevcut trendin gücünü ve yönünü gösterir.

| Skor | Anlam | Yön |
|------|-------|-----|
| 0-3 | Zayıf momentum | ↓ Düşüş |
| 4-6 | Nötr | → Yatay |
| **7-10** | **Güçlü momentum** | ↑ Yükseliş |

**Tespit yöntemi:** SMA hizalaması + RSI + 20 günlük fiyat değişimi

---

### 🤖 6. Algoritmik Akım İzi (Algorithmic Footprint)
> **HFT (High Frequency Trading)** ve **bot aktivitelerinin** piyasadaki izini sürer.

| Skor | Anlam | Etiket |
|------|-------|--------|
| 0-3 | İnsan ağırlıklı işlem | `HMN` |
| 3-5 | Karışık | `MIX` |
| 5-7 | Bot aktivitesi | `BOT` |
| **7-10** | **Güçlü HFT aktivitesi** | `HFT` |

**Tespit ettiği 4 bileşen:**
1. **Ritmik Tekrarlar:** Bar uzunluklarının algoritmik benzerliği
2. **Fraktal Yoğunluk:** Benzer boyutlu barların tekrarı
3. **Mikro Sıkışma→Patlama→Sürükleme:** HFT'nin tipik 3 aşamalı davranışı
4. **Hacim-Fiyat Senkronizasyonu:** Mükemmel korelasyon = bot

---

### 🐳 7. Arz Emilimi (Supply Absorption)
> **Kurumsal alıcıların**, gelen satışları emdiği güçlü destek bölgeleri.

| Skor | Emoji | Anlam |
|------|-------|-------|
| 0-3 | ⚪ | Emilim yok |
| 3-5.5 | 🟡 | Zayıf emilim |
| 5.5-7.5 | 🟠 | Orta seviye emilim |
| **7.5-10** | 🔴 | **Güçlü kurumsal emilim - Yükseliş sinyali** |

**Tespit yöntemi:** Uzun üst gölge + küçük gövde + yüksek hacim + fiyat stabilitesi + dar aralık kümelenmesi

---

### 🧊 8. Gizli Likidite (Iceberg Order)
> Büyük emirlerin parçalara bölünerek gizlendiği **buzdağı emirlerini** tespit eder.

| Skor | Tür | Anlam |
|------|-----|-------|
| < 5 | `Weak-Buy/Sell` | Zayıf iceberg sinyali |
| 5-7 | `Moderate-Buy/Sell` | Orta seviye |
| **≥ 7** | `Strong-Buy/Sell` | **Güçlü gizli emir - Büyük oyuncu aktif** |

**Tespit yöntemi:** Çok dar ATR (dar barlar) + yönlü kapanış + destek/dirence "duvara çarpma" + düşük hacimli ters hareket

---

## 📁 Çıktı Örnekleri

### Konsol Çıktısı

```
🔍 125 CSV dosyası bulundu. Analiz başlıyor...
========================================================================================================================
✅   1. THYAO                   | A: 7.2 D: 2.1 P: 1.5 F: 2.8 M: 8.4↑ Algo:6.3(BOT) Em:7.8🔴 Ice:7.2(Strong-Buy-Iceberg) | F:   285.50 Δ:+1.24%
✅   2. ASELS                   | A: 5.8 D: 3.4 P: 2.1 F: 4.5 M: 6.1→ Algo:4.2(MIX) Em:4.5🟡 Ice:3.8(Weak-Buy)           | F:    78.20 Δ:-0.50%
...
```

### Excel Raporu
Her analiz için tarih-saat damgalı bir Excel dosyası oluşturulur:
- `analiz_sonuclari_tumveri_20261028_143025.xlsx`
- `analiz_sonuclari_son90gun_20261028_143025.xlsx`

### Grafikler
`plots/` klasöründe otomatik oluşturulur:
- `top10_all_scores.png` (Tüm skorların kombine grafiği)
- `top10_Akümülasyon.png`
- `top10_Algoritmik_Akım.png`
- ... (her skor için ayrı grafik)

---

## 🗂️ Proje Yapısı

```
.
├── stock_analyzer.py        # Ana analiz kodu
├── README.md                # Bu dosya
├── data/                    # CSV verileri
│   ├── THYAO.csv
│   ├── ASELS.csv
│   └── ...
├── plots/                   # Oluşturulan grafikler (otomatik)
│   ├── top10_all_scores.png
│   └── ...
└── output/                  # Excel raporları (otomatik)
    └── analiz_sonuclari_*.xlsx
```

---

## 🎯 Yorumlama Rehberi (Nasıl Kullanılır?)

### Güçlü AL Sinyali Kombinasyonu
```
✅ Akümülasyon > 7
✅ Arz Emilimi > 7.5 (🔴)
✅ Momentum > 7 (↑)
✅ Iceberg = Strong-Buy
✅ Pump-Dump < 3 (temiz)
```

### Güçlü SAT / Uzak Dur Sinyali
```
⚠️ Dağıtım > 7
⚠️ Pump-Dump > 7
⚠️ Fake Breakout > 7
⚠️ Momentum < 3 (↓)
⚠️ Iceberg = Strong-Sell
```

### Manipülasyon Riski Yüksek
```
⚠️ Pump-Dump > 7 + Fake Breakout > 7
⚠️ Algoritmik Akım = HFT (çok aktif bot)
```

---

## ⚠️ Yasal Uyarı

Bu araç **yalnızca eğitim ve araştırma amaçlıdır**. Hiçbir şekilde **yatırım tavsiyesi değildir**. Tüm yatırım kararlarınızı kendi araştırmanıza ve lisanslı finansal danışmanınıza danışarak vermelisiniz.

Finansal piyasalarda geçmiş performans, gelecek sonuçların garantisi değildir.

---

## 🛠️ Teknik Detaylar

- **Dil:** Python 3.8+
- **Bağımlılıklar:** pandas, numpy, matplotlib, openpyxl
- **Minimum Veri:** Analiz için en az 30 günlük veri gereklidir (önerilen: 50+)
- **Performans:** 1000+ hisse için ~2-5 dakika (verinin büyüklüğüne bağlı)

---

## 🤝 Katkıda Bulunma

Katkılarınız memnuniyetle karşılanır! Lütfen:
1. Fork'layın
2. Feature branch açın (`git checkout -b feature/yeniOzellik`)
3. Commit'leyin (`git commit -m 'Yeni özellik eklendi'`)
4. Push'layın (`git push origin feature/yeniOzellik`)
5. Pull Request açın

---

## 📝 Lisans

MIT License - Detaylar için `LICENSE` dosyasına bakın.

---

## 📬 İletişim

Sorularınız için Issues bölümünü kullanabilirsiniz.

---

<div align="center">

**⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**

Made with ❤️ and Python

</div>
