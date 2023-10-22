import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Set direktori dataset
image_dir = '/content/images/'
annotation_dir = '/content/annotations/'

# Membagi dataset menjadi data pelatihan dan validasi
image_files = os.listdir(image_dir)
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

# Set label objek yang ingin dideteksi
label = 'object'

# Inisialisasi objek anotasi
annotation = tf.train.Example()

# Looping setiap gambar dalam dataset
for image_file in image_files:
    # Baca gambar
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    
    # Baca file anotasi (jika sudah ada)
    annotation_path = os.path.join(annotation_dir, image_file[:-4] + '.xml')
    if os.path.exists(annotation_path):
        with open(annotation_path, 'rb') as f:
            annotation.ParseFromString(f.read())
    
    # Looping setiap objek yang ingin dideteksi
    for obj in annotation.features.feature['object'].list:
        xmin = int(obj.int64_list.value[0])
        ymin = int(obj.int64_list.value[1])
        xmax = int(obj.int64_list.value[2])
        ymax = int(obj.int64_list.value[3])
        
        # Gambar kotak pada objek yang ingin dideteksi
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    # Tampilkan gambar
    cv2.imshow('image', image)
    cv2.waitKey(0)
    
    # Looping setiap objek yang ingin dideteksi pada gambar
    for obj in annotation.features.feature['object'].list:
        xmin = int(obj.int64_list.value[0])
        ymin = int(obj.int64_list.value[1])
        xmax = int(obj.int64_list.value[2])
        ymax = int(obj.int64_list.value[3])
        
        # Hitung koordinat objek relatif
        x_rel = (xmin + xmax) / (2.0 * width)
        y_rel = (ymin + ymax) / (2.0 * height)
        w_rel = (xmax - xmin) / float(width)
        h_rel = (ymax - ymin) / float(height)
        
        # Simpan anotasi ke dalam objek protobuf
        feature = annotation.features.feature['object'].list.add()
        feature.int64_list.value.extend([xmin, ymin, xmax, ymax])
        feature.bytes_list.value.extend([label.encode('utf8')])
        feature.float_list.value.extend([x_rel, y_rel, w_rel, h_rel])
    
    # Simpan anotasi ke dalam file XML
    annotation_path = os.path.join(annotation_dir, image_file[:-4] + '.xml')
    with open(annotation_path, 'wb') as f:
        f.write(annotation.SerializeToString())