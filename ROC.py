import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc

# Load trained Haar Cascade model
haar_cascade = cv2.CascadeClassifier('path/to/haar_cascade.xml')

# Load positive dataset (objek yang ingin dideteksi)
positive_images = ['path/to/positive_image1.jpg', 'path/to/positive_image2.jpg', ...]

# Load negative dataset (bukan objek yang ingin dideteksi)
negative_images = ['path/to/negative_image1.jpg', 'path/to/negative_image2.jpg', ...]

# Prepare labels
labels = np.concatenate([np.ones(len(positive_images)), np.zeros(len(negative_images))])

# Prepare data for ROC curve
scores = []
for image_path in positive_images + negative_images:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    objects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    scores.append(len(objects))

scores = np.array(scores)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
