import cv2
import numpy as np
from tensorflow.keras.models import load_model

print("1. Yapay Zeka Beyni (Model) Yükleniyor...")
model = load_model('v1/asl_model.h5')

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

    frame = cv2.flip(frame, 1)

    x1, y1, x2, y2 = 100, 100, 350, 350
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    el_goruntusu = frame[y1:y2, x1:x2]

    gri_el = cv2.cvtColor(el_goruntusu, cv2.COLOR_BGR2GRAY)
    kucuk_el = cv2.resize(gri_el, (28, 28))
    
    islenmis_el = kucuk_el / 255.0
    islenmis_el = islenmis_el.reshape(1, 28, 28, 1)

    tahminler = model.predict(islenmis_el, verbose=0)
    en_yuksek_ihtimal_indeksi = np.argmax(tahminler)
    
    tahmin_edilen_harf = harf_bul(en_yuksek_ihtimal_indeksi)

    cv2.putText(frame, f"Tahmin: {tahmin_edilen_harf}", (100, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Modelin Gordugu", kucuk_el)
    cv2.imshow("Gercek Zamanli ASL Algilayici", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()