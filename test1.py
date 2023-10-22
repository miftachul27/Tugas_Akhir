import cv2
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
  
img = cv2.imread("img10.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  

stop_data = cv2.CascadeClassifier('cars.xml')
  
found = stop_data.detectMultiScale(img_gray, minSize =(20, 20))
amount_found = len(found)
if amount_found != 0:
    for (x, y, width, height) in found:
              cv2.rectangle(img_rgb, (x, y), (x + height, y + width), (0, 255, 0), 5)

filename = 'savedImage.jpg'
cv2.imwrite(filename, img_rgb)