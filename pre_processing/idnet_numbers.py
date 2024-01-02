import cv2
import numpy as np

# Načítanie obrázka
image_path = '/home/vovo/Documents/DevOps/Complete_2022_DS_ML_Bootcamp/SEDS/final/0/0_4.jpg'

image = cv2.imread(image_path)
if image is None:
    print(f"Cannot load image from {image_path}")
    exit()

minimálna_plocha = 410

# Konverzia na šedotónový obraz a binarizácia
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Detekcia kontúr
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Pre vizualizáciu detekovaných kontúr
contour_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)
cv2.imshow('Detekované kontúry', contour_img)

# Filtrácia a spracovanie kontúr
"""
for contour in contours:
    if cv2.contourArea(contour) > minimálna_plocha:
        x, y, w, h = cv2.boundingRect(contour)
        number_roi = thresh[y:y+h, x:x+w]
        cv2.imshow('Výsledok', number_roi)
        cv2.waitKey(0)

"""
cv2.destroyAllWindows()
