import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

print("1. Veriler yükleniyor ve hazirlaniyor...")
# Bir önceki adımda yazdığımız hazırlık kodlarının aynısı
train_df = pd.read_csv('data/sign_mnist_train.csv')
y = train_df['label'].values
x = train_df.drop('label', axis=1).values
x = x / 255.0
x = x.reshape(-1, 28, 28, 1)

print("2. Veri 'Eğitim' ve 'Test' olarak ikiye ayriliyor...")
# Verinin %20'sini deneme sınavı (validation) için saklıyoruz
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

print("3. CNN Modeli inşa ediliyor...")
# Sıralı (Sequential) bir model oluşturuyoruz
model = Sequential([
    # GÖZ KISMI (Özellik Çıkarımı)
    # Resimdeki çizgileri, parmak kıvrımlarını tespit eden katman
    Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    # Gereksiz detayları atıp resmi özetleyen/küçülten katman
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    # BEYİN KISMI (Karar Verme)
    # Matrisi düz, tek boyutlu bir listeye çeviriyoruz
    Flatten(),
    # Yüksek kapasiteli düşünme nöronları
    Dense(128, activation='relu'),
    # Çıkış katmanı: 25 farklı harf ihtimali (A'dan Y'ye kadar, J hariç) için softmax
    Dense(25, activation='softmax') 
])

# Modeli derliyoruz (Hata hesaplama ve öğrenme stratejisini belirliyoruz)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\n--- İŞTE YAPAY ZEKANIN ANATOMİSİ (MODEL ÖZETİ) ---")
model.summary()
print("\n4. Model Eğitimi Basliyor (Bu işlem birkaç dakika sürebilir)...")
# Modeli eğitiyoruz. 'epochs=10' demek, modelin tüm veriyi 10 kere baştan sona okuyup çalışması demektir.
history = model.fit(x_train, y_train, 
                    epochs=10, 
                    batch_size=32, 
                    validation_data=(x_val, y_val))

print("\n5. Eğitim Tamamlandi! Model Kaydediliyor...")
# Öğrenilen bu beyni, daha sonra kamerada kullanmak üzere bir dosya olarak kaydediyoruz.
model.save('asl_model.h5')
print("Tebrikler! Beyin 'asl_model.h5' adiyla ana klasöre kaydedildi.")