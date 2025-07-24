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

max_features = None
ngram = (1,1)  #digerleri sabit tutulan kosularda en yuksek accuracy degerine sahip
C_values = [0.01, 0.1, 1.0, 10.0, 100.0]

for C in C_values:
    
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram, max_features=max_features)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)


    model = LogisticRegression(C=C, max_iter=200) #kullanacagim modeli olusturdum._
    model.fit(X_train_vectorized, y_train) #train ile modeli egit (fit)


    y_pred = model.predict(X_test_vectorized)
    acc = accuracy_score(y_test, y_pred)

    with mlflow.start_run():
        mlflow.log_param("ngram_range", ngram)
        mlflow.log_param("C", C)  
        mlflow.log_param("max_features", None)  # default
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

print(f"C={C}, accuracy={acc:.4f}")