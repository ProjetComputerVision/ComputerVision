import cv2 as cv
import numpy as np

cap = cv.VideoCapture('videos/Mousse.mp4')  # Remplacez par le chemin de votre vidéo

if not cap.isOpened():
    print("Erreur lors de l'ouverture de la vidéo")
    exit()

# Plages de couleur pour détecter le rouge en HSV//COool
lower_red1 = np.array([0, 50, 50])  # Rouge clair
upper_red1 = np.array([30, 255, 255])

lower_red2 = np.array([140, 50, 50])  # Rouge foncé
upper_red2 = np.array([180, 255, 255])

# Filtre morphologique pour réduire le bruit
kernel = np.ones((5, 5), np.uint8)

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

    # Appliquer un filtre morphologique pour réduire le bruit
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Trouver les contours dans le masque
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Dessiner les contours et détecter les objets
    for contour in contours:
        if cv.contourArea(contour) < 500:
            continue  # Ignorer les petits contours

        # Trouver le cercle minimal englobant l'objet
        (x, y), radius = cv.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius * 0.6)  # Redimensionner légèrement le cercle

        # Dessiner le cercle autour de l'objet (balle)
        cv.circle(frame, center, radius, (0, 255, 0), 2)


        # Calculer le centroïde du contour
        M = cv.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Centre en rouge

    # Afficher le masque
    cv.imshow("Masque de la couleur rouge", mask)

    # Afficher l'image avec la détection de l'objet
    cv.imshow("Détection de l'objet rouge", frame)

    # Quitter avec la touche 'q'
    if cv.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
