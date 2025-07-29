# AI Feedback – Gün 7

## İşe Yarayan Öneriler
``` 
- `predict.py` içindeki tahmin süreci `predict_message()` fonksiyonu olarak ayrıldı → test edilebilirlik arttı.
- Örnek test senaryoları (spam, ham, empty) AI tarafından önerildi ve `pytest` ile başarıyla çalıştı.
- Kod yapısı sadeleştirildi, main odaklı yapı yerine modüler yapı teşvik edildi.
```

## Güvenmediğim / Kontrol Ettiklerim
```
- AI, `"pred_samples.json` yazımı sırasında dosya kilitlenmesi için önlem almadığımı" söyledi ancak ben böyle bir öneri istememiştim.
- Bu, öneri kalitesini sorgulamam gerektiğini gösterdi → önerileri elle kontrol ettim.
- AI'nin hata tespitleri zaman zaman eksik (örneğin `best_model.pkl` eksikliği) → branch değiştirme veya dosya kontrolünü ben manuel yaptım.
```

## Genel Sonuç
```
Kod asistanı, hızlı prototipleme ve test önerilerinde oldukça faydalı. Ancak güvenlik ve dosya yönetimi gibi konularda dikkatli kontrol şart.
```