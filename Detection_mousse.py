import cv2 as cv
import numpy as np

cap = cv.VideoCapture('videos/Mousse.mp4')  # Remplacez par le chemin de votre vidéo

if not cap.isOpened():
    print("Erreur lors de l'ouverture de la vidéo")
    exit()

# Plages de couleur pour détecter le rouge en HSV
lower_red1 = np.array([0, 50, 50])  # Rouge clair
upper_red1 = np.array([30, 255, 255])

lower_red2 = np.array([140, 50, 50])  # Rouge foncé
upper_red2 = np.array([180, 255, 255])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image en HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Créer les masques pour le rouge clair et le rouge foncé
    mask_r1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv.inRange(hsv, lower_red2, upper_red2)
    mask = mask_r1 + mask_r2  # Combiner les deux masques

    # Appliquer le masque sur l'image originale
    result = cv.bitwise_and(frame, frame, mask=mask)

    # Afficher le masque seul
    cv.imshow('Masque de la couleur rouge', mask)

    # Afficher l'image d'origine avec le masque appliqué
    cv.imshow('Image avec le masque appliqué', result)
#lkjkn
    # Quitter avec la touche 'q'
    if cv.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
