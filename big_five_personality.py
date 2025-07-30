from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data-final.csv', sep='\t')

# gerekli sütunları seç
ext_cols = [f'EXT{i}' for i in range(1, 11)]
neu_cols = [f'EST{i}' for i in range(1, 11)]
df_sub = df[ext_cols + neu_cols].copy()
df_sub.dropna(inplace=True)

# hedef değişkeni oluştur
df_sub['EXT_score'] = df_sub[ext_cols].mean(axis=1)
threshold = df_sub['EXT_score'].mean()
df_sub['EXT_label'] = (df_sub['EXT_score'] > threshold).astype(int)

# girdi ve hedef belirle (neu soruları üzerinden ext tahmini)
X = df_sub[neu_cols]
y = df_sub['EXT_label']

# veri setini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# random forest modeli eğit
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_rf)

# classification report ve confusion matrix
report = classification_report(y_test, y_pred_rf, output_dict=True)
cm = confusion_matrix(y_test, y_pred_rf)

# özellik önemleri
importances = rf_model.feature_importances_
feature_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)

accuracy, report, cm, feature_importance

print(f"accuracy: {accuracy:.2f}\n")

print("classification report:")
for label, metrics in report.items():
    if isinstance(metrics, dict):
        print(f"  class {label}:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value:.2f}")
print()

print("confusion matrix:")
print(cm)

print("\nfeature importances:")
print(feature_importance)
