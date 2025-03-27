import cv2
import numpy as np
import os
import math
from scipy.optimize import curve_fit

# Créer le dossier de sortie s'il n'existe pas
output_dir = "Videos_Trajectoire"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Liste des vidéos à traiter
videos = ["Homographie_Mousse.mp4", "Homographie_Rugby.mp4", "Homographie_Tennis.mp4"]
video_dir = "Videos_Homographie"

# Plages de couleur pour la détection
lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([30, 255, 255])
lower_red2, upper_red2 = np.array([140, 50, 50]), np.array([180, 255, 255])
lower_color_bleu, upper_color_bleu = np.array([60, 125, 20]), np.array([140, 255, 255])
lower_color_jaune, upper_color_jaune = np.array([20, 50, 100]), np.array([40, 255, 255])

# Kernel pour filtrage
kernel = np.ones((5, 5), np.uint8)

# Facteurs de conversion
# Tableau de 200x120 cm dans une vidéo de 1280x720 pixels
PIXEL_TO_CM_X = 200 / 1280
PIXEL_TO_CM_Y = 120 / 720
PIXEL_TO_METER = (PIXEL_TO_CM_X + PIXEL_TO_CM_Y) / 2 / 100  # Moyenne en mètres


# Fonction pour ajuster une parabole
def parabolic_model(x, a, b, c):
    return a * x ** 2 + b * x + c


def fit_parabola(centers):
    if len(centers) < 3:
        return None, None

    x_points = np.array([p[0] for p in centers])
    y_points = np.array([p[1] for p in centers])

    try:
        # Ajuster une parabole aux points y = f(x)
        params, _ = curve_fit(parabolic_model, x_points, y_points)
        return params, x_points
    except:
        return None, None


def calculate_speed(centers, fps):
    if len(centers) < 2:
        return 0, 0, 0

    # Calculer les distances entre points consécutifs
    distances = []
    for i in range(1, len(centers)):
        dx = centers[i][0] - centers[i - 1][0]
        dy = centers[i][1] - centers[i - 1][1]
        distance = np.sqrt(dx ** 2 + dy ** 2)
        distances.append(distance)

    # Vitesse moyenne en pixels par frame
    avg_speed_pixels_per_frame = np.mean(distances)

    # Conversion en pixels par seconde
    speed_pixels_per_second = avg_speed_pixels_per_frame * fps

    # Conversion en m/s
    speed_m_per_second = speed_pixels_per_second * PIXEL_TO_METER

    # Conversion en km/h
    speed_km_per_hour = speed_m_per_second * 3.6

    return speed_pixels_per_second, speed_m_per_second, speed_km_per_hour


def calculate_realtime_speed(prev_center, current_center, fps):
    if prev_center is None or current_center is None:
        return 0, 0

    dx = current_center[0] - prev_center[0]
    dy = current_center[1] - prev_center[1]
    distance_pixels = np.sqrt(dx ** 2 + dy ** 2)

    # Vitesse en m/s
    speed_m_per_second = distance_pixels * PIXEL_TO_METER * fps

    # Vitesse en km/h
    speed_km_per_hour = speed_m_per_second * 3.6

    return speed_m_per_second, speed_km_per_hour


for video in videos:
    print(f"Traitement de {video}...")
    cap = cv2.VideoCapture(os.path.join(video_dir, video))
    if not cap.isOpened():
        print(f"Erreur lors de l'ouverture de {video}")
        continue

    # Obtenir les propriétés de la vidéo
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_time = 1.0 / fps

    # Créer l'objet VideoWriter pour la sortie
    output = cv2.VideoWriter(
        os.path.join(output_dir, f"Trajectoire_{video}"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    centers = []
    all_frames = []

    # Première passe: collecter tous les centres
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        all_frames.append(frame.copy())

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

        best_contour = None
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500 and area > max_area:
                max_area = area
                best_contour = contour

        if best_contour is not None:
            (x, y), radius = cv2.minEnclosingCircle(best_contour)
            center = (int(x), int(y))
            centers.append(center)

    # Calculer la vitesse moyenne
    _, avg_speed_mps, avg_speed_kmh = calculate_speed(centers, fps)
    print(f"Vitesse moyenne: {avg_speed_mps:.2f} m/s, {avg_speed_kmh:.2f} km/h")

    # Calculer l'angle initial et la vitesse initiale
    angle_initial = None
    vitesse_initiale = None

    if len(centers) >= 2:
        dx = (centers[1][0] - centers[0][0]) * PIXEL_TO_METER
        dy = (centers[1][1] - centers[0][1]) * PIXEL_TO_METER
        distance = math.sqrt(dx ** 2 + dy ** 2)
        vitesse_initiale = distance / frame_time
        angle_initial = math.degrees(math.atan2(-dy, dx))  # Négatif car y augmente vers le bas

    print(f"Angle initial: {angle_initial:.2f} degrés")
    print(f"Vitesse initiale: {vitesse_initiale:.2f} m/s")

    # Ajuster une parabole aux points collectés
    params, x_points = fit_parabola(centers)

    # Deuxième passe: dessiner la trajectoire parabolique
    frame_idx = 0
    while frame_idx < len(all_frames):
        frame = all_frames[frame_idx]

        # Calculer la vitesse en temps réel
        current_speed_mps = 0
        current_speed_kmh = 0

        if frame_idx > 0 and frame_idx < len(centers):
            prev_center = centers[frame_idx - 1] if frame_idx - 1 < len(centers) else None
            current_center = centers[frame_idx] if frame_idx < len(centers) else None

            if prev_center is not None and current_center is not None:
                current_speed_mps, current_speed_kmh = calculate_realtime_speed(prev_center, current_center, fps)

        # Dessiner les points détectés jusqu'à l'image actuelle
        for i, center in enumerate(centers[:frame_idx + 1]):
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # Dessiner la trajectoire réelle (points reliés) jusqu'à l'image actuelle
        for i in range(1, min(frame_idx + 1, len(centers))):
            cv2.line(frame, centers[i - 1], centers[i], (255, 0, 0), 2)

        # Dessiner la parabole ajustée seulement jusqu'à l'image actuelle
        if params is not None and frame_idx > 5:  # Attendre quelques frames pour avoir assez de points
            # Trouver le point x maximum atteint à ce moment
            current_max_x = max([p[0] for p in centers[:frame_idx + 1]]) if centers[:frame_idx + 1] else 0
            current_min_x = min([p[0] for p in centers[:frame_idx + 1]]) if centers[:frame_idx + 1] else 0

            # Dessiner la parabole seulement jusqu'au point actuel
            x_range = np.linspace(current_min_x, current_max_x, 100)

            prev_point = None
            for x in x_range:
                y = parabolic_model(x, *params)
                point = (int(x), int(y))

                if prev_point is not None:
                    cv2.line(frame, prev_point, point, (0, 255, 255), 2)
                prev_point = point

            # Prédiction de la trajectoire future (visible à chaque frame)
            future_x = np.linspace(current_max_x, width + 200, 100)  # Étendre au-delà de l'image
            prev_point = (int(current_max_x), int(parabolic_model(current_max_x, *params)))

            for x in future_x:
                y = parabolic_model(x, *params)
                point = (int(x), int(y))

                # Dessiner même si le point sort de l'image
                if prev_point is not None:
                    cv2.line(frame, prev_point, point, (0, 255, 0), 2)
                prev_point = point

                # Arrêter si on sort trop de l'image vers le bas
                if y > height + 100:
                    break

        # Afficher les informations de vitesse
        cv2.putText(frame, f"Vitesse moyenne: {avg_speed_mps:.2f} m/s", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Vitesse moyenne: {avg_speed_kmh:.1f} km/h", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Afficher l'angle initial et la vitesse initiale
        if angle_initial is not None and vitesse_initiale is not None:
            cv2.putText(frame, f"Vitesse initiale: {vitesse_initiale:.2f} m/s", (50, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Angle initial: {angle_initial:.2f} degres", (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Écrire l'image dans le fichier de sortie
        output.write(frame)
        frame_idx += 1

        # Afficher la progression
        if frame_idx % 10 == 0:
            print(f"Traitement de l'image {frame_idx}/{len(all_frames)}")

    # Libérer les ressources
    cap.release()
    output.release()

    print(f"Traitement terminé pour {video}")

print("Traitement terminé pour toutes les vidéos.")
