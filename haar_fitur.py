import cv2

# Fungsi untuk menghitung nilai fitur Haar-like pada gambar
def calculate_haar_like_features(image):
    # Konversi gambar ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Inisialisasi classifier Haar Cascade (misalnya: Tepi Horizontal)
    # haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'cars.xml')

    # Deteksi fitur Haar-like
    haar_features = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Mengembalikan hasil deteksi
    return haar_features

# Load gambar
image = cv2.imread('path/to/your/img100.jpg')

# Hitung nilai fitur Haar-like
features = calculate_haar_like_features(image)

# Tampilkan hasil deteksi pada gambar
for (x, y, w, h) in features:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Tampilkan gambar dengan deteksi
cv2.imshow('Haar-like Features', image)
cv2.waitKey(0)
cv2.destroyAllWindows()