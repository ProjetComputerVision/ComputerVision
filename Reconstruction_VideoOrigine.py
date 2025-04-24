import numpy as np
import cv2
import os


def superposer_videos(video_original_path, video_kalman_path, output_path):
    """
    Superpose la vidéo Kalman sur la vidéo originale en déformant précisément
    la vidéo Kalman pour l'adapter aux coordonnées spécifiées.
    """
    # Ouvrir les vidéos
    cap_original = cv2.VideoCapture(video_original_path)
    cap_kalman = cv2.VideoCapture(video_kalman_path)

    if not cap_original.isOpened() or not cap_kalman.isOpened():
        print(f"Erreur lors de l'ouverture des vidéos")
        return

    # Obtenir les propriétés des vidéos
    width_original = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_original = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width_kalman = int(cap_kalman.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_kalman = int(cap_kalman.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_original.get(cv2.CAP_PROP_FPS)

    # Points source dans la vidéo Kalman (coins de la vidéo Kalman)
    pts_source = np.array([
        [0, 0],  # coin supérieur gauche
        [width_kalman - 1, 0],  # coin supérieur droit
        [width_kalman - 1, height_kalman - 1],  # coin inférieur droit
        [0, height_kalman - 1]  # coin inférieur gauche
    ], dtype=np.float32)

    # Points destination dans la vidéo originale (coordonnées spécifiées)
    pts_destination = np.array([
        [123, -143],  # Coin supérieur gauche
        [673, 177],  # Coin supérieur droit
        [655, 435],  # Coin inférieur droit
        [24, 373]  # Coin inférieur gauche
    ], dtype=np.float32)

    # Créer une toile plus grande pour gérer les coordonnées négatives
    # Calculer le décalage nécessaire
    y_offset = int(abs(min(0, np.min(pts_destination[:, 1])))) + 10
    x_offset = int(abs(min(0, np.min(pts_destination[:, 0])))) + 10

    # Ajuster les points de destination pour le décalage
    pts_destination_adjusted = pts_destination.copy()
    pts_destination_adjusted[:, 1] += y_offset
    pts_destination_adjusted[:, 0] += x_offset

    # Dimensions de la toile élargie
    canvas_width = width_original + 2 * x_offset
    canvas_height = height_original + 2 * y_offset

    # Calculer la matrice de transformation perspective
    M = cv2.getPerspectiveTransform(pts_source, pts_destination_adjusted)

    # Créer l'objet VideoWriter pour la sortie
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width_original, height_original))

    print(f"Superposition de {os.path.basename(video_kalman_path)} sur {os.path.basename(video_original_path)}...")

    while True:
        ret_original, frame_original = cap_original.read()
        ret_kalman, frame_kalman = cap_kalman.read()

        if not ret_original or not ret_kalman:
            break

        # Créer une toile élargie
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Copier l'image originale sur la toile élargie
        canvas[y_offset:y_offset + height_original, x_offset:x_offset + width_original] = frame_original

        # Appliquer la transformation perspective sur la frame Kalman
        warped_kalman = cv2.warpPerspective(frame_kalman, M, (canvas_width, canvas_height))

        # Créer un masque pour les zones non-noires de l'image transformée
        gray_warped = cv2.cvtColor(warped_kalman, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_warped, 10, 255, cv2.THRESH_BINARY)

        # Créer le masque inverse
        mask_inv = cv2.bitwise_not(mask)

        # Extraire l'arrière-plan de la toile
        background = cv2.bitwise_and(canvas, canvas, mask=mask_inv)

        # Extraire les éléments de la vidéo Kalman transformée
        foreground = cv2.bitwise_and(warped_kalman, warped_kalman, mask=mask)

        # Combiner les deux images
        combined = cv2.add(background, foreground)

        # Extraire la région correspondant à l'image originale
        result = combined[y_offset:y_offset + height_original, x_offset:x_offset + width_original]

        # Écrire le résultat dans le fichier de sortie
        out.write(result)

        # Afficher le résultat
        cv2.imshow('Résultat', cv2.resize(result, (800, 450)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer les ressources
    cap_original.release()
    cap_kalman.release()
    out.release()
    cv2.destroyAllWindows()


def traiter_toutes_les_videos():
    # Créer le dossier de sortie s'il n'existe pas
    output_dir = "Videos_Reconstruction"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Définir les dossiers et les vidéos
    original_dir = "Videos"
    kalman_dir = "Videos_Kalman"

    videos = ["Mousse.mp4", "Rugby.mp4", "Tennis.mp4"]

    for video in videos:
        original_path = os.path.join(original_dir, video)
        kalman_path = os.path.join(kalman_dir, f"Kalman_Homographie_{video}")
        output_path = os.path.join(output_dir, f"Reconstruction_{video}")

        if not os.path.exists(original_path) or not os.path.exists(kalman_path):
            print(f"Erreur: Fichier manquant pour {video}")
            continue

        print(f"\nTraitement de {video}...")
        superposer_videos(original_path, kalman_path, output_path)

    print("Traitement terminé")


if __name__ == "__main__":
    traiter_toutes_les_videos()
