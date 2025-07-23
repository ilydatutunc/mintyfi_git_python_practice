import pandas as pd
data = pd.read_csv('sms_clean.csv', sep=',',encoding='utf-8')

X=data['message']
y=data['label']

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, 
    test_size=0.2,           # %20 si valis kalan %80 train
    random_state=42,         # rastgelelik sabitlensin
    stratify=y               # sınıf dagılımının korunması icin
)

print("Train class distribution:\n", y_train.value_counts(normalize=True))
print("Validation class distribution:\n", y_valid.value_counts(normalize=True))

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')

X_tfidf = vectorizer.fit_transform(X_train)  # train verisiyle tf-idf vektörleştirici eğit
X_valid_tfidf = vectorizer.transform(X_valid)  # valid verisini trainden öğrenilenle vektörleştir


print(f"train: {X_tfidf.shape}, valid: {X_valid_tfidf.shape}")

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Multinomial NB': MultinomialNB(),
    'Random Forest': RandomForestClassifier()
}

for name, model in models.items():
    print(f"\nModel: {name}")
    model.fit(X_tfidf, y_train)
    y_pred = model.predict(X_valid_tfidf)
    
    print("Accuracy:", accuracy_score(y_valid, y_pred))
    print("Precision:", precision_score(y_valid, y_pred, pos_label='spam'))
    print("Recall:", recall_score(y_valid, y_pred, pos_label='spam'))
    print("F1 Score:", f1_score(y_valid, y_pred, pos_label='spam'))
    print("Confusion Matrix:\n", confusion_matrix(y_valid, y_pred))

