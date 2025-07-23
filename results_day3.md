# Model Karşılaştırması Sonuçları

## Veri Seti Sınıf Dağılımı

- **Train seti**:
  - ham: %87.38
  - spam: %12.62

- **Validation seti**:
  - ham: %87.33
  - spam: %12.67

- Train veri şekli: (4135, 7414)  
- Validation veri şekli: (1034, 7414)

---

## Modellerin Performansları

| Model               | Accuracy | Precision (spam) | Recall (spam) | F1 Score (spam) | Confusion Matrix           |
|---------------------|----------|------------------|---------------|-----------------|----------------------------|
| Logistic Regression  | 0.9565   | 0.9778           | 0.6718        | 0.7964          | [[901, 2], [43, 88]]       |
| Multinomial NB      | 0.9671   | 1.0000           | 0.7405        | 0.8509          | [[903, 0], [34, 97]]       |
| Random Forest        | 0.9768   | 1.0000           | 0.8168        | 0.8992          | [[903, 0], [24, 107]]      |

---

## Kısa Yorum

- **Random Forest** modeli, genel doğruluk ve spam tespitinde en yüksek performansa sahiptir.
- **Precision** değerleri üç modelde de çok yüksek, yanlış pozitif oranı çok düşüktür.
- **Recall** açısından Logistic Regression spam tespiti konusunda diğer modellerden geride kalmıştır.
- Bu nedenle, spam mesaj algılama için **Random Forest** modeli tercih edilmelidir.

---

