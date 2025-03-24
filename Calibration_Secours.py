#Script de secours pour la calibration de la caméra

import cv2
import numpy as np
import glob

# 📌 Taille du damier (11x8)
nb_colonnes = 10  # Nombre de coins internes en largeur
nb_lignes = 7  # Nombre de coins internes en hauteur

# 📌 Critères d'arrêt pour l'algorithme de détection des coins
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 📌 Coordonnées des coins du damier en 3D
objp = np.zeros((nb_lignes * nb_colonnes, 3), np.float32)
objp[:, :2] = np.mgrid[0:nb_colonnes, 0:nb_lignes].T.reshape(-1, 2)

# 📌 Stockage des points 3D et 2D
objpoints = []  # Points 3D dans le monde réel
imgpoints = []  # Points 2D dans l'image

# 📌 Chargement des images (mets ici le bon chemin vers tes images)
images = glob.glob('/Users/corentinjozwiak/Documents/eseo/E4/COMPUTER VISION/Fichiers - checkboard-20250324/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 📌 Détection des coins du damier
    ret, corners = cv2.findChessboardCorners(gray, (nb_colonnes, nb_lignes), None)

    if ret:
        objpoints.append(objp)

        # 📌 Affinage des coordonnées des coins
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 📌 Affichage des coins détectés
        img = cv2.drawChessboardCorners(img, (nb_colonnes, nb_lignes), corners2, ret)
        cv2.imshow('Détection des Coins', img)

        # ⏸ Attendre que l'utilisateur appuie sur une touche pour continuer
        print(f"✅ Coins détectés sur {fname}. Appuie sur une touche pour continuer...")
        cv2.waitKey(0)  # Attente infinie

    else:
        print(f"⚠️ Échec de la détection des coins sur {fname}")

cv2.destroyAllWindows()

# 📌 Calibration de la caméra
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 📌 Affichage des résultats
print("\n🔹 Matrice intrinsèque :\n", mtx)
print("\n🔹 Coefficients de distorsion :\n", dist)

# 📌 Sauvegarde des paramètres
np.savez("calibration_data.npz", mtx=mtx, dist=dist)

print("\n✅ Calibration terminée ! Paramètres enregistrés dans 'calibration_data.npz'.")