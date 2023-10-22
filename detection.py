# import the necessary packages
import cv2
import numpy as np
import time

print(cv2.__version__)
detec = []
def pega_centro(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx, cy
offset=4
pos_line=200
car=0
cascade_src = 'goods.xml'
# video_src = 'dataset/video1.avi'
video_src = 'dataset/aw.mp4'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)



def video_frame():
    cap = cv2.VideoCapture(video_src)
    ret, frame = cap.read()  # read the camera frame
    return frame

def gen_frames():
    while True:
        frame = video_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

