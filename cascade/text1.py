import cv2
import numpy as np
import os
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import precision_score, recall_score, f1_score

class CascadeClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, cascade_path):
        self.cascade_path = cascade_path
        self.cascade = cv2.CascadeClassifier(self.cascade_path)

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        predictions = []
        for image in X:
            # Detect vehicles using the cascade classifier
            vehicles = self.cascade.detectMultiScale(image)
            predictions.append(1 if len(vehicles) > 0 else 0)
        return np.array(predictions)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

positive_image_dir = 'C:/Users/Prawira/Documents/Project/Joki/ta2hafizh/Latest Hapid/p2'
negative_image_dir = 'C:/Users/Prawira/Documents/Project/Joki/ta2hafizh/Latest Hapid/n2'
# Positive samples (vehicles)
pos_samples = []  # List of paths to positive sample images

# Negative samples (non-vehicles)
neg_samples = []  # List of paths to negative sample images

for filename in os.listdir(positive_image_dir):
    img = os.path.join(positive_image_dir, filename)
    if img is not None:
        pos_samples.append(img)

# Read negative images
for filename1 in os.listdir(negative_image_dir):
    img = os.path.join(negative_image_dir, filename1)
    if img is not None:
        neg_samples.append(img)

# Load the pre-trained Haar cascade classifier for vehicle detection
cascade_path = '19junes.xml'  # Path to the cascade XML file
cascade = cv2.CascadeClassifier(cascade_path)

# Create the training data and labels
train_data = []
train_labels = []

for pos_sample in pos_samples:
    image = cv2.imread(pos_sample, 0)  # Read the image as grayscale
    # Detect vehicles using the cascade classifier
    vehicles = cascade.detectMultiScale(image)
    for (x, y, w, h) in vehicles:
        roi = image[y:y + h, x:x + w]
        train_data.append(roi)
        train_labels.append(1)  # 1 for positive samples

# Process negative samples
for neg_sample in neg_samples:
    image = cv2.imread(neg_sample, 0)  # Read the image as grayscale
    # Detect vehicles using the cascade classifier
    vehicles = cascade.detectMultiScale(image)
    for (x, y, w, h) in vehicles:
        roi = image[y:y + h, x:x + w]
        train_data.append(roi)
        train_labels.append(0)  # 0 for negative samples


# Ensure consistent shape for all data
max_shape = max([data.shape for data in train_data])
train_data = np.array([data if data.shape == max_shape else cv2.resize(data, max_shape[:2]) for data in train_data])
train_labels = np.array(train_labels)

# Define the cross-validation strategy
cross_val = 5  # Number of folds for cross-validation

# Perform cross-validation
kf = KFold(n_splits=cross_val)
precision_scores = []
recall_scores = []
f1_scores = []

for train_index, test_index in kf.split(train_data):
    X_train, X_test = train_data[train_index], train_data[test_index]
    y_train, y_test = train_labels[train_index], train_labels[test_index]

    # Create an instance of the wrapper class
    cascade_wrapper = CascadeClassifierWrapper(cascade_path)

    # Fit the model and make predictions
    cascade_wrapper.fit(X_train, y_train)
    y_pred = cascade_wrapper.predict(X_test)

    # Calculate evaluation metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

# Print the average evaluation scores
print(f"Cross-Validation Precision: {np.mean(precision_scores):.2f}")
print(f"Cross-Validation Recall: {np.mean(recall_scores):.2f}")
print(f"Cross-Validation F1-Score: {np.mean(f1_scores):.2f}")
