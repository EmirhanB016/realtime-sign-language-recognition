import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Veri seti yükleniyor lütfen bekleyin...")

veri = pd.read_csv("data/sign_mnist_train.csv")

ilk_satir = veri.iloc[0]

etiket = ilk_satir["label"]
pikseller = ilk_satir[1:].values

resim = pikseller.reshape(28, 28).astype(float)

plt.imshow(resim, cmap = "gray")
plt.title(f"Sistemin Gordugu - Etiket: {etiket} (0=A, 1=B, 2=C...)")
plt.axis("off")
plt.show()