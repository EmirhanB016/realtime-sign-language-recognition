import cv2
import csv
import numpy as np

dosya_yolu = "data/hareketler.csv"
genislik, yukseklik = 600, 600

BAGLANTILAR = [
    (0, 1), (1, 2), (2, 3), (3, 4),      
    (0, 5), (5, 6), (6, 7), (7, 8),      
    (5, 9), (9, 10), (10, 11), (11, 12), 
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) 
]

print("Veri inceleme başladı. Sonraki satır için 'D' tuşuna basın, çıkmak için 'Q'.")

with open(dosya_yolu, mode='r') as f:
    okuyucu = csv.reader(f)
    
    for satir_no, satir in enumerate(okuyucu, start=1):
        if not satir: continue
        
        etiket = satir[0]
        koordinatlar = list(map(float, satir[1:]))
        
        tuval = np.zeros((yukseklik, genislik, 3), dtype=np.uint8)
        
        noktalar = []
        for i in range(0, len(koordinatlar), 3):
            x = int(koordinatlar[i] * genislik)
            y = int(koordinatlar[i+1] * yukseklik)
            noktalar.append((x, y))
            
        for baglanti in BAGLANTILAR:
            b_baslangic = noktalar[baglanti[0]]
            b_bitis = noktalar[baglanti[1]]
            cv2.line(tuval, b_baslangic, b_bitis, (0, 255, 0), 2) # Yeşil çizgiler
            
        for n in noktalar:
            cv2.circle(tuval, n, 5, (0, 0, 255), -1) # Kırmızı noktalar
            
        cv2.putText(tuval, f"Satir: {satir_no} | Harf: {etiket}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Veri Seti Iskelet Kontrolu", tuval)
        
        tus = cv2.waitKey(0) & 0xFF
        if tus == ord('q'):
            break

cv2.destroyAllWindows()