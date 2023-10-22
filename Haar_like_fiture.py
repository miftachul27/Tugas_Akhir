import cv2

# Load image
img = cv2.imread('img10.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define Haar-like feature for vehicle detection
vehicle_feature = cv2.CascadeClassifier('cars.xml')

# Detect vehicles using Haar-like feature
vehicles = vehicle_feature.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around detected vehicles
for (x, y, w, h) in vehicles:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    print(x,y)
    cv2.imshow("Integral Image",  cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2))
    
# # Extract Haar-like feature from detected face
# for (x, y, w, h) in vehicles:
#     # Extract region of interest
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]

#     # Calculate Haar-like feature
#     vehicle_features = vehicle_feature.detectMultiScale(gray)

#     # Print Haar-like feature
#     # print(vehicle_features)