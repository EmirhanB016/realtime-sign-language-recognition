import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Veri seti yükleniyor lütfen bekleyin...")

# Veriyi "data" klasörünün içinden okuyoruz
veri = pd.read_csv("data/sign_mnist_train.csv")

# İlk satırdaki veriyi alıyoruz
ilk_satir = veri.iloc[0]

# İlk sütun harfin ne olduğu (etiket), geri klan 784 sütun piksel değerleridir
etiket = ilk_satir["label"]
pikseller = ilk_satir[1:].values

# 784 uzunluğundaki düz sayı listesini 28x28 boyutunda bir resim matrisine çeviriyoruz
resim = pikseller.reshape(28, 28).astype(float)

plt.imshow(resim, cmap = "gray")
plt.title(f"Sistemin Gordugu - Etiket: {etiket} (0=A, 1=B, 2=C...)")
plt.axis("off")
plt.show()