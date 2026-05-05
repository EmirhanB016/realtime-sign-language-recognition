import pandas as pd
import numpy as np

print("1. Adim: Veri CSV dosyasindan okunuyor")
train_df = pd.read_csv("data/sign_mnist_train.csv")

# 1. X ve Y ayrımı
# Y (Cevaplar/Etiketler): Sadece "label" sütununu alıyoruz
y_train = train_df["label"].values

# X (Sorular/Görüntüler): "label" sütununu atıp geri kalan tüm pikselleri alıyoruz
x_train = train_df.drop("label", axis=1).values

print("Adim 2: Normalizasyon yapiliyor...")
# 2. Normalizasyon
# Pikseller 0-255 arasında. Yapay zeka 0 ile 1 arasındaki küçük sayıları daha hızlı öğrenir
# Bu yüzden tüm matrisi 255'e bölüyoruz
x_train = x_train / 255.0

print("Adim 3: Görüntüler CNN icin yeniden boyutlandiriliyor...")
# 3. CNN İÇİN YENİDEN BOYUTLANDIRMA (RESHAPE)
# Düz bir sayı listesi olan 784 pikseli -> 28x28 boyutunda bir fotoğrafa çeviriyoruz.
# Sondaki "1" rakamı resmin Siyah-Beyaz (tek kanallı) olduğunu temsil eder. 
# Renkli olsaydı (RGB) burası 3 olacaktı.
x_train = x_train.reshape(-1, 28, 28, 1)

print("\n--- İŞLEM TAMAMLANDI ---")
print(f"Eğitilecek Toplam Resim Sayisi ve Boyutlari (X): {x_train.shape}")
print(f"Eğitilecek Toplam Etiket Sayisi (Y): {y_train.shape}")