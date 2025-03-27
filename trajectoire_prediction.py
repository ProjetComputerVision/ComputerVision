import cv2
import numpy as np
import os

# Liste des vidéos à traiter
videos = ["Homographie_Mousse.mp4", "Homographie_Rugby.mp4", "Homographie_Tennis.mp4"]
video_dir = "Videos_Homographie"

# Facteur de conversion pixels -> mètres (ajuste cette valeur selon l'échelle réelle)
pixel_to_meter = 2/1280 # Ex: 1 pixel = 1 cm (0.01 m)

# Nombre de frames pour lisser la vitesse
window_size = 5

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

    fps = cap.get(cv2.CAP_PROP_FPS)  # Récupérer le framerate de la vidéo
    frame_time = 1 / fps  # Temps entre chaque image
    centers = []
    velocities = []  # Liste des vitesses pour lisser la mesure

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

        # Dessiner la trajectoire réelle en BLEU
        for i in range(1, len(centers)):
            cv2.line(frame, centers[i - 1], centers[i], (255, 0, 0), 2)

        # Ajustement et affichage de la parabole sur la trajectoire observée
        if len(centers) > 8:
            pts = np.array(centers[-15:], dtype=np.float32)  # Utiliser les 15 derniers points
            x_vals, y_vals = pts[:, 0], pts[:, 1]

            # Ajustement d'une parabole (modèle quadratique)
            coeffs = np.polyfit(x_vals, y_vals, 2)  # y = ax^2 + bx + c
            poly = np.poly1d(coeffs)

            # Générer des points pour tracer la parabole observée
            x_fit = np.linspace(min(x_vals), max(x_vals), num=50)
            y_fit = poly(x_fit)

            # Clipper les valeurs pour rester dans l'image
            height, width, _ = frame.shape
            x_fit = np.clip(x_fit, 0, width - 1)
            y_fit = np.clip(y_fit, 0, height - 1)

            # Dessiner la parabole ajustée en ROUGE
            for i in range(len(x_fit) - 1):
                cv2.line(frame,
                         (int(x_fit[i]), int(y_fit[i])),
                         (int(x_fit[i + 1]), int(y_fit[i + 1])),
                         (0, 0, 255), 2, cv2.LINE_AA)

        # Calcul précis de la vitesse en m/s
        if len(centers) > window_size:
            total_distance = 0
            total_time = (window_size - 1) * frame_time  # Temps total sur 5 frames

            for i in range(-window_size + 1, 0):  # Parcours des 5 dernières positions
                dx = centers[i][0] - centers[i - 1][0]
                dy = centers[i][1] - centers[i - 1][1]
                distance_pixels = np.sqrt(dx**2 + dy**2)  # Distance en pixels
                total_distance += distance_pixels * pixel_to_meter  # Conversion en mètres

            # Vitesse moyenne sur les 5 dernières frames
            speed = total_distance / total_time
            velocities.append(speed)

            # Moyenne sur les dernières vitesses pour lisser
            if len(velocities) > 5:
                velocities.pop(0)  # Supprimer la plus ancienne valeur
            avg_speed = np.mean(velocities)

            # Affichage de la vitesse sur l'écran
            cv2.putText(frame, f"Vitesse: {avg_speed:.2f} m/s", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Prédiction de la trajectoire future (modèle quadratique)
        if len(centers) > 8:
            x_pred = np.linspace(x_vals[-1], x_vals[-1] + 500, num=50)
            y_pred = poly(x_pred)

            # Clipper les valeurs pour rester dans l'image
            x_pred = np.clip(x_pred, 0, width - 1)
            y_pred = np.clip(y_pred, 0, height - 1)

            # Dessiner la trajectoire prédite en VERT
            for i in range(len(x_pred) - 1):
                cv2.line(frame,
                         (int(x_pred[i]), int(y_pred[i])),
                         (int(x_pred[i + 1]), int(y_pred[i + 1])),
                         (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Tracking avec vitesse, modèle et prédiction', frame)
        if cv2.waitKey(90) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

print("Traitement terminé.")
