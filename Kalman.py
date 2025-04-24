import cv2
import numpy as np
import os

# Liste des vidéos à traiter
videos = ["Homographie_Mousse.mp4", "Homographie_Rugby.mp4", "Homographie_Tennis.mp4"]
video_dir = "Videos_Homographie"
output_dir = "Videos_Kalman"  # Dossier de sortie pour les vidéos traitées

# Création du dossier de sortie s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# Plages de couleur pour la détection
lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([30, 255, 255])
lower_red2, upper_red2 = np.array([140, 50, 50]), np.array([180, 255, 255])
lower_color_bleu, upper_color_bleu = np.array([60, 125, 20]), np.array([140, 255, 255])
lower_color_jaune, upper_color_jaune = np.array([20, 50, 100]), np.array([40, 255, 255])

# Kernel pour filtrage
kernel = np.ones((5, 5), np.uint8)

# Initialisation du filtre Kalman avec vélocité, accélération, et la gravité
kalman = cv2.KalmanFilter(6, 2)
# Matrice de transition ajustée pour une meilleure modélisation de la gravité
kalman.transitionMatrix = np.array([[1, 0, 1, 0, 0.5, 0],
                                    [0, 1, 0, 1, 0, 0.5],
                                    [0, 0, 1, 0, 1, 0],
                                    [0, 0, 0, 1, 0, 1],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]], np.float32)

kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0]], np.float32)

# Ajustement des paramètres de bruit pour une trajectoire plus réaliste
kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
kalman.errorCovPost = np.eye(6, dtype=np.float32)

# Variables pour la détection et la prédiction
last_center = None
detected = False
frames_since_last_detection = 0
max_prediction_frames = 30  # Augmentation du nombre de frames pour la prédiction

# Historique des positions pour tracer la trajectoire
trajectory_history = []
max_trajectory_points = 20

for video in videos:
    print(f"Traitement de {video}...")
    cap = cv2.VideoCapture(os.path.join(video_dir, video))
    if not cap.isOpened():
        print(f"Erreur lors de l'ouverture de {video}")
        continue

    # Réinitialisation des variables pour chaque vidéo
    last_center = None
    detected = False
    frames_since_last_detection = 0
    trajectory_history = []

    # Récupérer les propriétés de la vidéo d'entrée pour le VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Créer le nom du fichier de sortie
    output_filename = os.path.join(output_dir, f"Kalman_{video}")

    # Définir le codec et créer l'objet VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    # Ajustement des paramètres du filtre Kalman selon le type de vidéo
    if "Tennis" in video or "Mousse" in video:
        # Pour les balles qui rebondissent moins, réduire l'accélération verticale
        kalman.statePost = np.zeros((6, 1), np.float32)
        # Ajout d'un biais de gravité plus faible pour ces objets
        kalman.transitionMatrix[1, 4] = 0.3  # Gravité plus faible
    else:
        # Pour le rugby, configuration standard
        kalman.statePost = np.zeros((6, 1), np.float32)
        kalman.transitionMatrix[1, 4] = 0.5  # Gravité standard

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

        significant_contours = [c for c in contours if cv2.contourArea(c) >= 500]

        if significant_contours:
            detected = True
            frames_since_last_detection = 0

            # Détection de la balle
            contour = max(significant_contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))

            # Rectangle autour de l'objet détecté
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # Mise à jour du filtre Kalman avec la mesure (position de la balle)
            measurement = np.array([center[0], center[1]], np.float32)
            kalman.correct(measurement)

            last_center = center

            # Ajout de la position à l'historique
            trajectory_history.append(center)
            if len(trajectory_history) > max_trajectory_points:
                trajectory_history.pop(0)
        else:
            frames_since_last_detection += 1
            detected = False

        # Prédiction de la position future du centroïde
        predicted = kalman.predict()
        predicted_center = (int(predicted[0].item()), int(predicted[1].item()))

        # Affichage de la prédiction actuelle
        if detected:
            # Lorsque la balle est détectée, afficher la prédiction à partir du centre réel
            cv2.circle(frame, last_center, 5, (0, 255, 0), -1)

            # Visualisation de la trajectoire prédite
            future_points = []
            temp_kalman = cv2.KalmanFilter(6, 2)
            temp_kalman.transitionMatrix = kalman.transitionMatrix.copy()
            temp_kalman.statePost = kalman.statePost.copy()

            # Prédire plusieurs points dans le futur
            for i in range(10):
                pred = temp_kalman.predict()
                pt = (int(pred[0].item()), int(pred[1].item()))
                future_points.append(pt)

            # Dessiner la trajectoire prédite
            for i in range(1, len(future_points)):
                cv2.line(frame, future_points[i - 1], future_points[i], (0, 255, 255), 1)

            # Flèche directionnelle depuis le centroïde réel
            cv2.arrowedLine(frame, last_center, predicted_center, (0, 0, 255), 2, tipLength=0.3)

        else:
            # Si la balle n'est plus détectée mais qu'on a une prédiction
            if last_center is not None and frames_since_last_detection <= max_prediction_frames:
                # Afficher la prédiction actuelle
                cv2.circle(frame, predicted_center, 5, (0, 255, 0), -1)

                # Visualisation de la trajectoire future prédite
                future_points = [predicted_center]
                temp_kalman = cv2.KalmanFilter(6, 2)
                temp_kalman.transitionMatrix = kalman.transitionMatrix.copy()
                temp_kalman.statePost = kalman.statePost.copy()

                # Prédire plusieurs points dans le futur
                for i in range(10):
                    pred = temp_kalman.predict()
                    pt = (int(pred[0].item()), int(pred[1].item()))
                    future_points.append(pt)

                # Dessiner la trajectoire prédite
                for i in range(1, len(future_points)):
                    cv2.line(frame, future_points[i - 1], future_points[i], (0, 255, 255), 1)

                # Flèche directionnelle basée sur la direction du mouvement prédit
                next_pred = future_points[1] if len(future_points) > 1 else None
                if next_pred:
                    cv2.arrowedLine(frame, predicted_center, next_pred, (0, 0, 255), 2, tipLength=0.3)

                # Ajouter le point prédit à l'historique de trajectoire pour continuité
                trajectory_history.append(predicted_center)
                if len(trajectory_history) > max_trajectory_points:
                    trajectory_history.pop(0)

                # Mettre à jour le dernier centre connu
                last_center = predicted_center

        # Affichage d'informations sur l'écran
        cv2.putText(frame, f"Detection: {'Oui' if detected else 'Non'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        # Écrire la frame dans le fichier de sortie
        out.write(frame)

        # Affichage
        cv2.imshow('Tracking avec Kalman et prédiction de trajectoire', frame)
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

    # Libérer les ressources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Vidéo enregistrée : {output_filename}")

print("Traitement terminé")
