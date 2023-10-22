import cv2
import numpy as np

# Membaca gambar
image = cv2.imread('img100.jpg', 0)  # Ubah 'path_to_image.jpg' dengan path gambar yang ingin digunakan

# Menghitung integral image
integral_image = np.cumsum(np.cumsum(image, axis=0), axis=1)

# Menampilkan integral image
print(integral_image)