import cv2
import numpy as np


cap = cv2.VideoCapture('videos/Rugby.mp4')  # Remplacez par le chemin de votre vidéo


if not cap.isOpened():
    print("Erreur lors de l'ouverture de la vidéo")
    exit()



lower_color_bleu = np.array([60, 125, 20])  # Plage basse de la couleur bleue
upper_color_bleu = np.array([140, 255, 255])
#lower_color_jaune = np.array([20, 50, 100])  # Plage basse de la couleur jaune
#upper_color_jaune = np.array([40, 255, 255])
while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color_bleu, upper_color_bleu)
    cv2.imshow('mask', mask)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) >= 5:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)


            if radius > 10:
                cv2.circle(frame, center, radius, (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                print(f"Centre de la balle : {center}")


    cv2.imshow('Détection de la balle', frame)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
