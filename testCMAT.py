import cv2

# Load trained Cascade Classifier
cascade_path = '19junes.xml'
cascade = cv2.CascadeClassifier(cascade_path)

# Path to the test images
test_images_path = 'C:\Users\Miftachul Hafizh\Desktop\revisi program\data testing'

# List of test image filenames
test_images = ['test1.jpg', 'test2.jpg', 'test3.jpg',]

# Initialize counters for true positives, false positives, and total objects
true_positives = 0
false_positives = 0
total_objects = 0

for image_filename in test_images:
    # Load the test image
    image_path = test_images_path + '/' + image_filename
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect objects using the trained cascade
    detected_objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw bounding boxes around detected objects
    for (x, y, w, h) in detected_objects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        total_objects += 1

    # Load ground truth information if available
    # Assuming ground truth is stored in a text file with the same name as the image
    ground_truth_path = test_images_path + '/' + image_filename.replace('.jpg', '.txt')
    with open(ground_truth_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            true_positives += 1

    # Display the image with detections
    cv2.imshow('Detection Result', img)
    cv2.waitKey(0)

# Calculate accuracy metrics
precision = true_positives / total_objects
recall = true_positives / len(test_images)

print(f'Precision: {precision}')
print(f'Recall: {recall}')

cv2.destroyAllWindows()