import numpy as np
import cv2
import os


def warp_video(video_path, output_path, output_size=(800, 480)):
    """
    Transforme une vidéo en vue de dessus en utilisant l'homographie calculée

    Args:
        video_path: Chemin vers la vidéo d'entrée
        output_path: Chemin pour la vidéo de sortie
        output_size: Taille de la vidéo de sortie (largeur, hauteur) en pixels
    """
    # Charger l'homographie
    try:
        data = np.load("Data/homography_data.npz")
        H = data['H']
        H_inverse = data['H_inverse']
        pts_monde_reel = data['pts_monde_reel']
        print("Données d'homographie chargées avec succès")
    except:
        print("Erreur: Impossible de charger les données d'homographie")
        return

    # Ouvrir la vidéo d'entrée
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la vidéo {video_path}")
        return

    # Obtenir les propriétés de la vidéo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Créer l'objet VideoWriter pour la sortie
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec pour MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    # Calculer la matrice de transformation pour la vue de dessus
    # Nous allons définir les dimensions de la vue de dessus en fonction des dimensions réelles
    max_x = np.max(pts_monde_reel[:, 0])
    max_y = np.max(pts_monde_reel[:, 1])

    # Facteur d'échelle pour convertir cm en pixels dans la vue de dessus
    scale_x = output_size[0] / max_x
    scale_y = output_size[1] / max_y

    # Matrice de mise à l'échelle
    S = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ])

    # Matrice de transformation finale (homographie suivie de mise à l'échelle)
    M = S @ H

    print(f"Transformation de la vidéo {video_path}...")
    print(f"Dimensions de sortie: {output_size[0]}x{output_size[1]} pixels")
    print(f"Nombre total d'images: {frame_count}")

    # Traiter chaque image de la vidéo
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Appliquer la transformation perspective
        warped = cv2.warpPerspective(frame, M, output_size)

        # Écrire l'image transformée dans la vidéo de sortie
        out.write(warped)

        # Afficher la progression
        frame_index += 1
        if frame_index % 100 == 0:
            print(f"Traitement: {frame_index}/{frame_count} images ({frame_index / frame_count * 100:.1f}%)")

        # Afficher les images (optionnel, ralentit le traitement)
        if frame_index % 30 == 0:  # Afficher seulement une image sur 30
            cv2.imshow('Original', frame)
            cv2.imshow('Vue de dessus', warped)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Libérer les ressources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Transformation terminée. Vidéo enregistrée sous {output_path}")


def process_all_videos():
    """
    Traite toutes les vidéos du dossier "Videos" et crée des versions transformées
    """
    # Créer le dossier de sortie s'il n'existe pas
    output_dir = "Videos_Homographie"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lister les vidéos dans le dossier d'entrée
    video_dir = "Videos"
    if not os.path.exists(video_dir):
        print(f"Erreur: Le dossier {video_dir} n'existe pas")
        return

    videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

    if not videos:
        print(f"Aucune vidéo trouvée dans le dossier {video_dir}")
        return

    # Traiter chaque vidéo
    for video in videos:
        input_path = os.path.join(video_dir, video)
        output_path = os.path.join(output_dir, f"Homographie_{video}")

        print(f"\nTraitement de {video}...")
        warp_video(input_path, output_path)


if __name__ == "__main__":
    process_all_videos()
