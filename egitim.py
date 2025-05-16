from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import pandas as pd
import pickle

df = pd.read_csv("veriseti.csv")
y = df["Etiket"]
X = df.drop("Etiket", axis=1)

# Eğitim ve test olarak 2'ye böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline oluştur
pipeline = Pipeline([
    ('rf', RandomForestClassifier(random_state=42))  # Rastgele Orman modeli
])

# Modeli eğit
pipeline.fit(X_train, y_train)

# Modelin test setindeki tahminlerini al
y_pred = pipeline.predict(X_test)

# Sonuçları değerlendir
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk Oranı = {accuracy}")

# Modeli kaydet
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)