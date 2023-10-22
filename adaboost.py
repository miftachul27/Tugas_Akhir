import cv2
import numpy as np

# Load the face cascade
vehicles_cascade = cv2.CascadeClassifier('cars.xml')

# Load the training data
training_data = cv2.imread('img10.jpg')

# Convert the training data to grayscale
gray = cv2.cvtColor(training_data, cv2.COLOR_BGR2GRAY)

# Train the classifier
vehicles = vehicles_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Create a list of positive samples

pos = []
for (x, y, w, h) in vehicles:
    pos.append(gray[y:y+h, x:x+w])

# Create a list of negative samples
neg = []
for i in range(100):
    x = int(np.random.rand() * (gray.shape[1] - 100))
    y = int(np.random.rand() * (gray.shape[0] - 100))
    neg.append(gray[y:y+100, x:x+100])

# Inisialisasi jumlah fitur terbaik dan jumlah putaran Adaboost
best_features = 10
num_rounds = 50

# Create a list of Haar-like features
features = []
for i in range(100):
   features.append(cv2.HaarFeature())

# Train the Adaboost classifier
adaboost = cv2.Adaboost()
adaboost.train(features, pos, neg, num_rounds=50)

# Get the best features
best_features = adaboost.getBestFeatures(10)