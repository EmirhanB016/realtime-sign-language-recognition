import cv2
import mediapipe as mp
import pickle
import numpy as np

model_yolu = "models/harf_modeli.pkl"
try:
    with open(model_yolu, "rb") as f:
        model = pickle.load(f)
    print("Yapay zeka modeli başarıyla yüklendi!")
except Exception as e:
    print("Model yüklenirken hata oluştu. Önce train.py dosyasını çalıştırdığınızdan emin olun.")
    exit()

mp_eller = mp.solutions.hands
mp_cizim = mp.solutions.drawing_utils

kamera = cv2.VideoCapture(0)
print("Kamera açılıyor... (Çıkmak için 'q' tuşuna basın)")

with mp_eller.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as eller:
    while True:
        ret, frame = kamera.read()
        if not ret: continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        sonuclar = eller.process(rgb_frame)
        
        if sonuclar.multi_hand_landmarks:
            for el_isaretleri in sonuclar.multi_hand_landmarks:
                mp_cizim.draw_landmarks(frame, el_isaretleri, mp_eller.HAND_CONNECTIONS)
                
                anlik_veri = []
                for lm in el_isaretleri.landmark:
                    anlik_veri.extend([lm.x, lm.y, lm.z])
                
                X_tahmin = np.array([anlik_veri])
                
                tahmin_edilen_harf = model.predict(X_tahmin)[0]
                
                cv2.putText(frame, f"Tahmin: {tahmin_edilen_harf}", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

        cv2.imshow("Gercek Zamanli Isaret Dili Tanima", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

kamera.release()
cv2.destroyAllWindows()