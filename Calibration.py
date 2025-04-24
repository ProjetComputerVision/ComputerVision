import numpy as np
import cv2
import glob

# Taille de l'échiquier
nb_corners_x = 10  # Nombre de coins internes sur l'axe X
nb_corners_y = 7  # Nombre de coins internes sur l'axe Y
pattern_size = (nb_corners_x, nb_corners_y)

# Préparation des points du monde réel
objp = np.zeros((nb_corners_x * nb_corners_y, 3), np.float32)
objp[:, :2] = np.mgrid[0:nb_corners_x, 0:nb_corners_y].T.reshape(-1, 2)

# Stockage des points
objpoints = []  # Points du monde réel
imgpoints = []  # Points détectés

# Charger les images
images = glob.glob("Checkboard/*.jpg")
print(f"Nombre d'images trouvées : {len(images)}")

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Erreur : Impossible de charger l'image {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Augmente le contraste

    # Détection des coins de l'échiquier avec différentes options
    ret, corners = cv2.findChessboardCorners(gray, pattern_size,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)
    print(f"Image {fname} - Échiquier détecté : {ret}")

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        cv2.imshow("Échiquier détecté", img)
        cv2.waitKey(500)  # Attendre 500 ms pour voir

cv2.destroyAllWindows()

# Vérification avant calibration
if len(objpoints) == 0 or len(imgpoints) == 0:
    print("Erreur : Aucun point détecté ! Vérifiez le damier et les images.")
    exit()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Affichage des résultats
print("\n=== Paramètres de calibration ===")
print("Matrice de la caméra :\n", mtx)
print("\nCoefficient de distorsion :\n", dist)
print("\nVecteurs de rotation :\n", rvecs)
print("\nVecteurs de translation :\n", tvecs)

# Sauvegarde
np.savez("Data/calibration_data.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
print("\nCalibration enregistrée dans calibration_data.npz")
