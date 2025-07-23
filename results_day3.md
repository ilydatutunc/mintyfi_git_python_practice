
- **Train set**:
  - ham: %87.38
  - spam: %12.62

- **Validation set**:
  - ham: %87.33
  - spam: %12.67

- Train data shape: (4135, 7414)  
- Validation data shape: (1034, 7414)

```

Modellerin Performansları

| Model                | Accuracy | Precision (spam) | Recall (spam) | F1 Score (spam) | Confusion Matrix           |
|--------------------- |----------|------------------|---------------|-----------------|----------------------------|
| Logistic Regression  | 0.9565   | 0.9778           | 0.6718        | 0.7964          | [[901, 2], [43, 88]]       |
| Multinomial NB       | 0.9671   | 1.0000           | 0.7405        | 0.8509          | [[903, 0], [34, 97]]       |
| Random Forest        | 0.9787   | 0.9910           | 0.8397        | 0.9091          | [[902, 1], [21, 110]]      |

```

- **Random Forest** modeli, genel doğruluk ve spam tespitinde en yüksek performansa sahiptir.
- Üç modelde de **Precision** çok yüksek, yani FP çok az.
- **Recall** açısından `Random Forest` en iyi sonucu verdi,spam mesajları daha iyi yakalıyor.
- Logistic Regression, spam tespiti konusunda diğer modellere göre biraz daha geride kaldı.
- Bu nedenle, spam mesaj algılamada **Random Forest (random_state=42)** modeli en uygunu.

---

