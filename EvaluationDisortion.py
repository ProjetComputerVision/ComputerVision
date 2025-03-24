# Charger une image de test
import cv2

from Calibration import mtx, dist, objpoints, imgpoints, rvecs, tvecs

test_img = cv2.imread("Checkboard/Checkboard (1).jpg")  # Utilisez une de vos images d'échiquier
h, w = test_img.shape[:2]

# Calculer la nouvelle matrice de caméra optimale
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Méthode 1: Correction avec undistort
dst = cv2.undistort(test_img, mtx, dist, None, newcameramtx)
# Découper la région d'intérêt
x, y, w, h = roi
dst_cropped = dst[y:y+h, x:x+w]

# Méthode 2: Correction avec remap (plus précise)
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
dst2 = cv2.remap(test_img, mapx, mapy, cv2.INTER_LINEAR)
dst2_cropped = dst2[y:y+h, x:x+w]

# Afficher les résultats
cv2.imshow('Image originale', test_img)
cv2.imshow('Image corrigée (undistort)', dst)
cv2.imshow('Image corrigée (remap)', dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Enregistrer les résultats
cv2.imwrite('original.jpg', test_img)
cv2.imwrite('corrected_undistort.jpg', dst)
cv2.imwrite('corrected_remap.jpg', dst2)

# Calcul de l'erreur de reprojection pour évaluer la qualité de la calibration
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print(f"Erreur de reprojection totale: {mean_error/len(objpoints)}")
print(f"Plus cette valeur est proche de zéro, meilleure est la calibration")
