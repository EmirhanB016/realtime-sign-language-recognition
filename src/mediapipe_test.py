import cv2
import mediapipe as mp

print("Kamera açılıyor... (Çıkmak için 'q' tuşuna basın)")

mp_eller = mp.solutions.hands
mp_cizim = mp.solutions.drawing_utils

kamera = cv2.VideoCapture(0)

with mp_eller.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as eller:
    while True:
        ret, frame = kamera.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sonuclar = eller.process(rgb_frame)
        
        if sonuclar.multi_hand_landmarks:
            for el_isaretleri in sonuclar.multi_hand_landmarks:
                # 1. İskeleti ekrana çiz
                mp_cizim.draw_landmarks(frame, el_isaretleri, mp_eller.HAND_CONNECTIONS)
                
                # 2. VERİ ÇEKİMİ: Sadece Bilek noktasının (0. Landmark) koordinatlarını al
                bilek_x = el_isaretleri.landmark[mp_eller.HandLandmark.WRIST].x
                bilek_y = el_isaretleri.landmark[mp_eller.HandLandmark.WRIST].y
                bilek_z = el_isaretleri.landmark[mp_eller.HandLandmark.WRIST].z
                
                # Terminale yazdır (Virgülden sonra 2 basamak olacak şekilde)
                print(f"Bilek -> X: {bilek_x:.2f} | Y: {bilek_y:.2f} | Z: {bilek_z:.2f}")
                
        cv2.imshow("MediaPipe Iskelet Testi", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

kamera.release()
cv2.destroyAllWindows()