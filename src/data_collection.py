import cv2
import mediapipe as mp
import csv
import os

# Ayarlar
isaret_adi = "merhaba"  # Kaydedilecek hareketin etiketi
dosya_yolu = "data/hareketler.csv"

# Klasör yoksa oluştur
if not os.path.exists("data"):
    os.makedirs("data")

mp_eller = mp.solutions.hands
mp_cizim = mp.solutions.drawing_utils
kamera = cv2.VideoCapture(0)

print(f"'{isaret_adi}' hareketi için veri toplanıyor. Kayıt için 's' tuşuna basın.")

with mp_eller.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as eller:
    while True:
        ret, frame = kamera.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sonuclar = eller.process(rgb_frame)
        
        if sonuclar.multi_hand_landmarks:
            for el_isaretleri in sonuclar.multi_hand_landmarks:
                mp_cizim.draw_landmarks(frame, el_isaretleri, mp_eller.HAND_CONNECTIONS)
                
                # 's' tuşuna basıldığında koordinatları kaydet
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    veri_satiri = [isaret_adi]
                    for lm in el_isaretleri.landmark:
                        veri_satiri.extend([lm.x, lm.y, lm.z])
                    
                    # CSV dosyasına ekle
                    with open(dosya_yolu, mode='a', newline='') as f:
                        yazici = csv.writer(f)
                        yazici.writerow(veri_satiri)
                    print(f"Kayıt başarılı! Toplam veri satırı eklendi.")

        cv2.imshow("Veri Toplama Merkezi", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

kamera.release()
cv2.destroyAllWindows()