import cv2
from sklearn.model_selection import KFold

# Load dataset and labels
dataset_pos = [...]  # List of positive image paths
dataset_neg = [...]  # List of negative image paths
labels_pos = [1] * len(dataset_pos)  # Positive labels
labels_neg = [0] * len(dataset_neg)  # Negative labels
dataset = dataset_pos + dataset_neg
labels = labels_pos + labels_neg

# Define Cascade Classifier
cascade_classifier = cv2.CascadeClassifier("19junes.xml")

# Perform cross-validation
kfold = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kfold.split(dataset):
    # Split dataset into train and test subsets
    train_data = [dataset[i] for i in train_index]
    train_labels = [labels[i] for i in train_index]
    test_data = [dataset[i] for i in test_index]
    test_labels = [labels[i] for i in test_index]

    # Train Cascade Classifier
    for image_path, label in zip(train_data, train_labels):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Train the classifier with positive and negative samples
        # ...

    # Test Cascade Classifier
    true_positives = 0
    false_positives = 0
    for image_path, label in zip(test_data, test_labels):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply the trained classifier to detect objects
        detections = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(detections) > 0:
            # Object detected
            if label == 1:
                true_positives += 1
            else:
                false_positives += 1

    # Calculate detection rate and false positive rate
    detection_rate = true_positives / len(test_data)
    false_positive_rate = false_positives / len(test_data)

    # Print the results
    print("Detection Rate:", detection_rate)
    print("False Positive Rate:", false_positive_rate)
