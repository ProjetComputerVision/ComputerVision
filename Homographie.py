import numpy as np
import cv2
import os

# Chargement des paramètres intrinsèques de la caméra (si nécessaire)
try:
    calib_data = np.load('calibration_data.npz')
    mtx = calib_data['mtx']
    dist = calib_data['dist']
    print("Paramètres de calibration chargés avec succès")
except:
    print("Paramètres de calibration non trouvés, utilisation sans correction de distorsion")
    mtx = None
    dist = None


def calculate_homography():
    # 1. Définir les coordonnées des 4 coins du tableau blanc en pixels (à modifier selon votre image)
    # Format: [coin supérieur gauche, coin supérieur droit, coin inférieur droit, coin inférieur gauche]
    pts_image = np.array([
        [123, -143],  # Coin supérieur gauche (x, y) en pixels
        [673, 177],  # Coin supérieur droit (x, y) en pixels
        [655, 435],  # Coin inférieur droit (x, y) en pixels
        [24, 373]  # Coin inférieur gauche (x, y) en pixels
    ], dtype=np.float32)

    # 2. Définir les coordonnées réelles correspondantes en cm
    # Le coin supérieur gauche est l'origine (0,0)
    pts_monde_reel = np.array([
        [0, 0],  # Coin supérieur gauche (0, 0) cm
        [200, 0],  # Coin supérieur droit (200, 0) cm
        [200, 120],  # Coin inférieur droit (200, 120) cm
        [0, 120]  # Coin inférieur gauche (0, 120) cm
    ], dtype=np.float32)

    # 3. Calculer la matrice d'homographie
    H, status = cv2.findHomography(pts_image, pts_monde_reel)
    H_inverse = np.linalg.inv(H)

    print("Matrice d'homographie calculée:")
    print(H)

    return H, H_inverse, pts_image, pts_monde_reel


def test_homography(video_path, H, H_inverse, pts_image, pts_monde_reel):
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la vidéo {video_path}")
        return

    # Lire la première frame
    ret, frame = cap.read()
    if not ret:
        print("Erreur: Impossible de lire la vidéo")
        cap.release()
        return

    # Afficher les coins sur l'image
    img_with_corners = frame.copy()
    for i, (x, y) in enumerate(pts_image):
        cv2.circle(img_with_corners, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(img_with_corners, f"P{i + 1}", (int(x) + 10, int(y) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Dessiner le contour du tableau blanc
    pts = pts_image.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(img_with_corners, [pts], True, (0, 255, 0), 2)

    # Afficher l'image avec les coins
    cv2.imshow("Image avec les coins", img_with_corners)

    # Test de conversion de coordonnées
    # Exemple: convertir un point au milieu de l'image en coordonnées réelles
    h, w = frame.shape[:2]
    test_point_image = np.array([[[w // 2, h // 2]]], dtype=np.float32)
    test_point_reel = cv2.perspectiveTransform(test_point_image, H)

    print(f"Point test dans l'image: ({w // 2}, {h // 2}) pixels")
    print(f"Point test dans le monde réel: ({test_point_reel[0][0][0]:.2f}, {test_point_reel[0][0][1]:.2f}) cm")

    # Créer une grille de points pour visualiser la transformation
    grid_size = 50  # Espacement en pixels
    grid_points_image = []
    for x in range(0, w, grid_size):
        for y in range(0, h, grid_size):
            grid_points_image.append([x, y])

    grid_points_image = np.array([grid_points_image], dtype=np.float32)
    grid_points_reel = cv2.perspectiveTransform(grid_points_image, H)

    # Dessiner la grille transformée sur une nouvelle image
    grid_img = frame.copy()
    for i, point in enumerate(grid_points_image[0]):
        x, y = point
        real_x, real_y = grid_points_reel[0][i]
        cv2.circle(grid_img, (int(x), int(y)), 2, (0, 255, 255), -1)

    cv2.imshow("Grille de points", grid_img)

    # Attendre que l'utilisateur appuie sur une touche
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()


def main():
    # Calculer l'homographie
    H, H_inverse, pts_image, pts_monde_reel = calculate_homography()

    # Sauvegarder l'homographie
    np.savez("homography_data.npz", H=H, H_inverse=H_inverse,
             pts_image=pts_image, pts_monde_reel=pts_monde_reel)
    print("Matrices d'homographie sauvegardées dans homography_data.npz")

    # Lister les vidéos dans le dossier "Videos"
    video_dir = "Videos"
    if not os.path.exists(video_dir):
        print(f"Erreur: Le dossier {video_dir} n'existe pas")
        return

    videos = [os.path.join(video_dir, f) for f in os.listdir(video_dir)
              if f.endswith(('.mp4', '.avi', '.mov'))]

    if not videos:
        print(f"Aucune vidéo trouvée dans le dossier {video_dir}")
        return

    # Tester l'homographie sur la première vidéo
    test_homography(videos[0], H, H_inverse, pts_image, pts_monde_reel)

    print("\nFonctions disponibles pour utiliser l'homographie:")
    print("1. image_to_world(point_image): Convertit des coordonnées de l'image en coordonnées du monde réel")
    print("2. world_to_image(point_reel): Convertit des coordonnées du monde réel en coordonnées de l'image")


def image_to_world(point_image, H=None):
    """
    Convertit des coordonnées de l'image en coordonnées du monde réel
    point_image: tuple (x, y) en pixels
    """
    if H is None:
        data = np.load("homography_data.npz")
        H = data['H']

    point = np.array([[[point_image[0], point_image[1]]]], dtype=np.float32)
    point_reel = cv2.perspectiveTransform(point, H)
    return point_reel[0][0]


def world_to_image(point_reel, H_inverse=None):
    """
    Convertit des coordonnées du monde réel en coordonnées de l'image
    point_reel: tuple (x, y) en cm
    """
    if H_inverse is None:
        data = np.load("homography_data.npz")
        H_inverse = data['H_inverse']

    point = np.array([[[point_reel[0], point_reel[1]]]], dtype=np.float32)
    point_image = cv2.perspectiveTransform(point, H_inverse)
    return point_image[0][0]


if __name__ == "__main__":
    main()
