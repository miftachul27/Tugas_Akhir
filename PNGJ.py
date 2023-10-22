import cv2
import numpy as np

# Load the trained model
model = cv2.CascadeClassifier('19junes.xml')

# Load the image for testing
image_path = 'uji.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform the detection
cars = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Ground truth (manually annotated)
true_positive = len(cars)  # Number of cars detected
false_negative = 40 - true_positive  # Total cars - Detected cars

false_positive = 0  # Initialize false positive count

# Draw rectangles around the detected cars
for (x, y, w, h) in cars:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Load the ground truth (e.g., from XML file)
ground_truth = [(x1, y1, w1, h1), (x2, y2, w2, h2), ...]  # List of tuples (x, y, width, height)

for (x, y, w, h) in ground_truth:
    detected = False
    for (cx, cy, cw, ch) in cars:
        # Check if the detected car box intersects with the ground truth box
        if (x < cx + cw and x + w > cx and y < cy + ch and y + h > cy):
            detected = True
            break
    if not detected:
        false_negative += 1

# Calculate True Negative
true_negative = 40 - (true_positive + false_negative)

# Calculate accuracy, precision, recall
accuracy = (true_positive + true_negative) / 40
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)

# Print the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Display the image with detections
cv2.imshow('Detected Cars', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
