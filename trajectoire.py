import cv2
import numpy as np
import os

# Liste des vidéos à traiter
videos = ["Homographie_Mousse.mp4", "Homographie_Rugby.mp4", "Homographie_Tennis.mp4"]
video_dir="Videos_Homographie"
# Plages de couleur pour la détection
lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([30, 255, 255])
lower_red2, upper_red2 = np.array([140, 50, 50]), np.array([180, 255, 255])
lower_color_bleu, upper_color_bleu = np.array([60, 125, 20]), np.array([140, 255, 255])
lower_color_jaune, upper_color_jaune = np.array([20, 50, 100]), np.array([40, 255, 255])

# Kernel pour filtrage
kernel = np.ones((5, 5), np.uint8)

for video in videos:
    print(f"Traitement de {video}...")
    cap = cv2.VideoCapture(os.path.join(video_dir, video))
    if not cap.isOpened():
        print(f"Erreur lors de l'ouverture de {video}")
        continue

    centers = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if "Mousse" in video:
            mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        elif "Rugby" in video:
            mask = cv2.inRange(hsv, lower_color_bleu, upper_color_bleu)
        elif "Tennis" in video:
            mask = cv2.inRange(hsv, lower_color_jaune, upper_color_jaune)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            centers.append(center)
            cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # Dessiner la trajectoire
        for i in range(1, len(centers)):
            cv2.line(frame, centers[i - 1], centers[i], (255, 0, 0), 2)

        cv2.imshow('Tracking', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

print("Traitement terminé.")