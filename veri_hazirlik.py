import pandas as pd
import numpy as np

print("1. Adim: Veri CSV dosyasindan okunuyor")
train_df = pd.read_csv("data/sign_mnist_train.csv")

y_train = train_df["label"].values

x_train = train_df.drop("label", axis=1).values

print("Adim 2: Normalizasyon yapiliyor...")
x_train = x_train / 255.0

print("Adim 3: Görüntüler CNN icin yeniden boyutlandiriliyor...")
x_train = x_train.reshape(-1, 28, 28, 1)

print("\n--- İŞLEM TAMAMLANDI ---")
print(f"Eğitilecek Toplam Resim Sayisi ve Boyutlari (X): {x_train.shape}")
print(f"Eğitilecek Toplam Etiket Sayisi (Y): {y_train.shape}")