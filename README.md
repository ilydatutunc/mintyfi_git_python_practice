# Big Five Kişilik Tahmini Projesi

## Proje Özeti
Bu proje, IPIP Big Five kişilik testi verisi kullanarak, Neuroticism anket maddelerine verilen cevaplardan yola çıkarak bireyin Extraversion seviyesini (yüksek/düşük) tahmin etmeyi amaçlamaktadır.

## Veri Seti Hakkında

Bu projede kullanılan veri seti, [Big Five Personality Test - Kaggle](https://www.kaggle.com/datasets/tunguz/big-five-personality-test/data) adresinden indirilmiştir. 

Veri, IPIP Big Five kişilik testi anketlerinden toplanan yüz binlerce yanıtı içerir ve kişilik özelliklerini beş temel boyutta (Extraversion, Neuroticism, Agreeableness, Conscientiousness, Openness) tanımlar.

## Veri Seti
Veri seti: IPIP Big Five kişilik testi (`data-final.csv`).

- **Girdi olarak:** Neuroticism (EST1–EST10) maddeleri kullanıldı.
- **Hedef değişken olarak:** Extraversion (EXT1–EXT10) maddelerinden oluşturulan etiket.

### Hedef Değişken
`EXT_label` olarak adlandırılan hedef:
- Extraversion ortalaması veri setinin genel ortalamasının üzerindeyse 1 (yüksek),
- Aksi halde 0 (düşük).

## Yöntem
- Eksik veriler temizlendi.
- Veri %80 eğitim, %20 test olarak ayrıldı.
- Random Forest sınıflandırma modeli eğitildi.
- Model performansı accuracy, classification report ve confusion matrix ile değerlendirildi.
- Ayrıca hangi Neuroticism maddelerinin tahmin için daha etkili olduğu belirlendi.

## Sonuçlar

| Ölçüt               | Değer  |
|---------------------|--------|
| Accuracy            | 0.84   |
| Precision (Class 0) | 0.86   |
| Recall (Class 0)    | 0.83   |
| F1 Score (Class 0)  | 0.84   |
| Precision (Class 1) | 0.84   |
| Recall (Class 1)    | 0.87   |
| F1 Score (Class 1)  | 0.86   |

### Confusion Matrix:

|              | Predicted 0 | Predicted 1 |
|--------------|-------------|-------------|
| **Actual 0** | 6173        | 1296        |
| **Actual 1** | 967         | 6592        |

## Öne Çıkan Özellik Önemleri

Extraversion tahmininde en etkili Neuroticism maddeleri:

| Madde Kodu | Önem Skoru |
|------------|------------|
| EST6       | 0.133      |
| EST2       | 0.108      |
| EST4       | 0.104      |
| EST9       | 0.099      |
| EST10      | 0.095      |
