import cv2
import numpy as np
import os

# Liste des vidéos à traiter
videos = ["Homographie_Mousse.mp4", "Homographie_Rugby.mp4", "Homographie_Tennis.mp4"]
video_dir = "Videos_Homographie"

# Facteur de conversion pixels -> mètres
pixel_to_meter = 2 / 1280

# Plages de couleur pour la détection
lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([30, 255, 255])
lower_red2, upper_red2 = np.array([140, 50, 50]), np.array([180, 255, 255])
lower_color_bleu, upper_color_bleu = np.array([60, 125, 20]), np.array([140, 255, 255])
lower_color_jaune, upper_color_jaune = np.array([20, 50, 100]), np.array([40, 255, 255])

# Kernel pour filtrage
kernel = np.ones((5, 5), np.uint8)

# Initialisation du filtre Kalman
kalman = cv2.KalmanFilter(4, 2)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1

# Variables pour suivre la dernière position de la balle
last_center = None

for video in videos:
    print(f"Traitement de {video}...")
    cap = cv2.VideoCapture(os.path.join(video_dir, video))
    if not cap.isOpened():
        print(f"Erreur lors de l'ouverture de {video}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps
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

            # Correction du filtre Kalman avec la mesure actuelle
            kalman.correct(np.array([x, y], np.float32))

            # Prédiction de la position future avec Kalman
            predicted = kalman.predict()
            predicted_center = (int(predicted[0]), int(predicted[1]))

            # Calcul de la direction à partir de la dernière position et de la position actuelle
            if last_center is not None:
                dx = center[0] - last_center[0]
                dy = center[1] - last_center[1]
                magnitude = np.sqrt(dx**2 + dy**2)
                if magnitude > 0:  # Si la magnitude est significative
                    direction = (dx / magnitude, dy / magnitude)  # Normalisation pour obtenir une direction unitaire

                    # Dessiner une flèche représentant la direction du mouvement de la balle
                    arrow_length = 100  # Longueur de la flèche
                    end_point = (int(center[0] + direction[0] * arrow_length),
                                int(center[1] + direction[1] * arrow_length))
                    cv2.arrowedLine(frame, center, end_point, (255, 0, 0), 2, tipLength=0.05)

            # Mettre à jour la dernière position de la balle
            last_center = center

            # Boîte de détection autour de la balle
            cv2.rectangle(frame, (int(x - radius), int(y - radius)), (int(x + radius), int(y + radius)), (0, 255, 255), 2)

        cv2.imshow('Tracking avec filtre Kalman et prediction de la direction', frame)
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

print("Traitement terminé")
