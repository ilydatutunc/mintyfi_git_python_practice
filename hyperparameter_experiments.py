import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression #bu modeli kullanicam
from sklearn.metrics import accuracy_score

#ikinci gün temizledigim dataset i okuyorum
df = pd.read_csv('sms_clean.csv', sep=',',encoding='utf-8')

# print(df.head()) --okundu mu kontrol ettim

X = df['message']
y = df['label']

# Train/valid split 80/20 oranında ayırdım
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

max_features_values = [1000, 2000, 3000, None]
ngram = (1,1)  #digerleri sabit tutulan kosularda en yuksek accuracy degerine sahip

for max_features in max_features_values:
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram, max_features=max_features)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    model = LogisticRegression(C=100.0, max_iter=200)
    model.fit(X_train_vectorized, y_train)

    
    y_pred = model.predict(X_test_vectorized)
    acc = accuracy_score(y_test, y_pred)

    
    with mlflow.start_run():
        mlflow.log_param("ngram_range", ngram)
        mlflow.log_param("C", 100.0) #en yüksek accuracy degerini veren C degeri
        mlflow.log_param("max_features", max_features)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

    print(f"max_features={max_features}, accuracy={acc:.4f}")
