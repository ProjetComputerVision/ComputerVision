import cv2
import numpy as np
import os

# Définition des plages de couleur pour la détection
# Rugby (bleu)
lower_color_bleu = np.array([60, 125, 20])
upper_color_bleu = np.array([140, 255, 255])

# Mousse (rouge)
lower_red1 = np.array([0, 50, 50])  # Rouge clair
upper_red1 = np.array([30, 255, 255])
lower_red2 = np.array([140, 50, 50])  # Rouge foncé
upper_red2 = np.array([180, 255, 255])

# Tennis (jaune)
lower_color_jaune = np.array([20, 50, 100])
upper_color_jaune = np.array([40, 255, 255])

# Dossier contenant les vidéos
video_dir = "Videos_Homographie"
output_dir = "Videos_Detection"

# Créer le dossier de sortie s'il n'existe pas
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Liste des vidéos à traiter
videos = ["Homographie_Mousse.mp4", "Homographie_Rugby.mp4", "Homographie_Tennis.mp4"]

# Filtre morphologique pour réduire le bruit
kernel = np.ones((5, 5), np.uint8)

for video in videos:
    print(f"Traitement de {video}...")

    # Ouvrir la vidéo
    cap = cv2.VideoCapture(os.path.join(video_dir, video))
    if not cap.isOpened():
        print(f"Erreur lors de l'ouverture de la vidéo {video}")
        continue

    # Obtenir les propriétés de la vidéo
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Créer les objets VideoWriter pour les sorties
    mask_output = cv2.VideoWriter(
        os.path.join(output_dir, f"masque_{video}"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    tracking_output = cv2.VideoWriter(
        os.path.join(output_dir, f"tracking_{video}"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    # Nouveau VideoWriter pour la trajectoire
    trajectory_output = cv2.VideoWriter(
        os.path.join(output_dir, f"trajectoire_{video}"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    frame_count = 0

    # Liste pour stocker les positions précédentes
    trajectory_points = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Créer une copie pour la trajectoire
        trajectory_frame = frame.copy()

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Traitement de l'image {frame_count}")

        # Convertir l'image en HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Détection de la couleur selon la vidéo
        if "Mousse" in video:
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask1 + mask2
        elif "Rugby" in video:
            mask = cv2.inRange(hsv, lower_color_bleu, upper_color_bleu)
        elif "Tennis" in video:
            mask = cv2.inRange(hsv, lower_color_jaune, upper_color_jaune)

        # Appliquer des filtres morphologiques pour réduire le bruit
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Appliquer un flou gaussien pour lisser les contours
        if "Rugby" in video:
            mask = cv2.GaussianBlur(mask, (15, 15), 0)

        # Trouver les contours dans le masque
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Variable pour stocker le centre de l'objet actuel
        current_center = None

        # Dessiner les contours et détecter les objets
        for contour in contours:
            # Ignorer les petits contours
            if cv2.contourArea(contour) < 500:
                continue

            # Trouver le cercle minimal englobant l'objet
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            current_center = center

            # Ajustement du rayon selon le type de balle
            if "Mousse" in video:
                radius = int(radius * 0.6)  # Redimensionner légèrement pour la mousse
            else:
                radius = int(radius)

            # Dessiner le cercle autour de l'objet
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.circle(trajectory_frame, center, radius, (0, 255, 0), 2)

            # Dessiner le centre de l'objet
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.circle(trajectory_frame, center, 5, (0, 0, 255), -1)

        # Ajouter le centre actuel à la liste des points de trajectoire
        if current_center is not None:
            trajectory_points.append(current_center)

        # Dessiner la trajectoire sur l'image de trajectoire
        if len(trajectory_points) > 1:
            # Dessiner des lignes entre les points consécutifs
            for i in range(1, len(trajectory_points)):
                cv2.line(trajectory_frame, trajectory_points[i - 1], trajectory_points[i], (255, 0, 0), 2)

        # Convertir le masque en BGR pour l'enregistrement
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Écrire dans les fichiers de sortie
        mask_output.write(mask_bgr)
        tracking_output.write(frame)
        trajectory_output.write(trajectory_frame)

    # Libérer les ressources
    cap.release()
    mask_output.release()
    tracking_output.release()
    trajectory_output.release()

    print(f"Traitement terminé pour {video}")

print("Traitement terminé pour toutes les vidéos.")
