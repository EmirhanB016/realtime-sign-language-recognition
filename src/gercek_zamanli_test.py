import cv2
import numpy as np
from tensorflow.keras.models import load_model

print("1. Yapay Zeka Beyni (Model) Yükleniyor...")
# Az önce eğittiğimiz modeli hafızaya alıyoruz
model = load_model('asl_model.h5')

# Harf sözlüğümüzü oluşturuyoruz. 
# Not: Amerikan İşaret Dilinde 'J' ve 'Z' harfleri hareket gerektirdiği için 
# statik resim veri setinde (MNIST) yer almaz. 
def harf_bul(etiket):
    if etiket >= 9:
        etiket += 1
    return chr(etiket + 65)

print("2. Kamera Başlatiliyor... (Kapatmak için ekrandayken 'q' tuşuna basin)")
kamera = cv2.VideoCapture(0)

while True:
    ret, frame = kamera.read()
    if not ret:
        continue

    # Kamerayı ayna gibi tersine çeviriyoruz (kullanımı daha kolay olsun diye)
    frame = cv2.flip(frame, 1)

    # İşlem yapılacak karenin (ROI - Region of Interest) koordinatları
    # Kamerada elinizi bu karenin içine koyacaksınız
    x1, y1, x2, y2 = 100, 100, 350, 350
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Sadece o karenin içindeki görüntüyü kesip alıyoruz
    el_goruntusu = frame[y1:y2, x1:x2]

    # Ön İşleme: Modelin anlayacağı formata (28x28 Siyah-Beyaz) çeviriyoruz
    gri_el = cv2.cvtColor(el_goruntusu, cv2.COLOR_BGR2GRAY)
    kucuk_el = cv2.resize(gri_el, (28, 28))
    
    # Arka planı siyah, eli beyaz yapmak algılamayı kolaylaştırır (Veri setimiz öyleydi)
    # Eğer kamerada ters çalışırsa, bu cv2.bitwise_not satırını silebilirsiniz
    #kucuk_el = cv2.bitwise_not(kucuk_el) 

    # Normalizasyon ve boyutlandırma
    islenmis_el = kucuk_el / 255.0
    islenmis_el = islenmis_el.reshape(1, 28, 28, 1)

    # Yapay zekadan tahmin alıyoruz
    tahminler = model.predict(islenmis_el, verbose=0)
    en_yuksek_ihtimal_indeksi = np.argmax(tahminler)
    
    # Çıkan sayıyı A, B, C gibi harflere dönüştürüyoruz
    tahmin_edilen_harf = harf_bul(en_yuksek_ihtimal_indeksi)

    # Sonucu ekrana yazdırıyoruz
    cv2.putText(frame, f"Tahmin: {tahmin_edilen_harf}", (100, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # İşlenmiş küçük resmi de kenarda görelim (Sistem tam olarak ne görüyor?)
    cv2.imshow("Modelin Gordugu", kucuk_el)
    cv2.imshow("Gercek Zamanli ASL Algilayici", frame)

    # 'q' tuşuna basılırsa döngüyü kır ve kamerayı kapat
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik işlemleri
kamera.release()
cv2.destroyAllWindows()