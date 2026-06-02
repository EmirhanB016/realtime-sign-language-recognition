import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

dosya_yolu = "data/hareketler.csv"
if not os.path.exists(dosya_yolu):
    print("Hata: data/hareketler.csv dosyası bulunamadı! Önce veri toplayın.")
    exit()

df = pd.read_csv(dosya_yolu, header=None)

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Eğitim veri seti boyutu: {X_train.shape[0]} örnek")
print(f"Test veri seti boyutu: {X_test.shape[0]} örnek")

print("\nModel eğitiliyor... (Bu işlem çok kısa sürecek)")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
basari_orani = accuracy_score(y_test, y_pred)
print(f"\nModel Başarı Oranı (Accuracy): %{basari_orani * 100:.2f}")

print("\nDetaylı Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
model_yolu = "models/harf_modeli.pkl"
with open(model_yolu, "wb") as f:
    pickle.dump(model, f)

print(f" Yapay zeka modeli '{model_yolu}' olarak başarıyla kaydedildi!")