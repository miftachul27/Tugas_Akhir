import cv2

# Load image
img = cv2.imread("uji.jpg", cv2.IMREAD_GRAYSCALE)

# Calculate integral image
int_img = cv2.integral(img)

print(int_img)

# Show integral image
cv2.imshow("Integral Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()