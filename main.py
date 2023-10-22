import cv2
import numpy as np
from detection import video_frame, gen_frames
from urllib import response
from flask import Flask, jsonify, make_response, request,render_template,url_for,redirect, Response
from flask_cors import CORS
import os
from os.path import join, dirname, realpath
import os
import csv
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage



DEBUG = True
# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)
app.static_folder = 'static'

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})
# Upload folder
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER
detec = []
def pega_centro(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx, cy


@app.route('/')
def main():
     return render_template('index.html')

@app.route('/dashboard')
def dashboard():
     return render_template('index.html')

@app.route('/deteksi')
def deteksi():
     return render_template('detection.html')

@app.route('/deteksi' , methods=['POST'])
def prosesdeteksi():
     f = request.files['file']
     f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
     print('display_video filename: ' + f.filename)
     res = "/".join((app.config['UPLOAD_FOLDER'], f.filename))
     # print('display_video filenameresresresresres ' + res)
     # filename = f.filename
     # print('filename  ' + filename)
     img = cv2.imread(res)
     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     threshold_value = 128
     _, thresholded_image = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)
     _, thresholded_image1 = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)
     img_thresh = cv2.cvtColor(thresholded_image1, cv2.COLOR_BGR2RGB)



    # Save the thresholded image
     # cv2.imwrite("thresholded_image.jpg", thresholded_image)

     stop_data = cv2.CascadeClassifier('19junes.xml')
     
     found = stop_data.detectMultiScale(img_gray, minSize =(20, 20))
     amount_found = len(found)
     car_count = 0
     if amount_found != 0:
         for (x, y, width, height) in found:
                   cv2.rectangle(img_rgb, (x, y), (x + height, y + width), (0, 255, 0), 2)
                   car_count += 1
     if amount_found != 0:
         for (x, y, width, height) in found:
                   cv2.rectangle(img_thresh, (x, y), (x + height, y + width), (0, 255, 0), 2)
     
     filename = f.filename
     file = "filename.jpg"
     cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'] , "filename.jpg"), img_rgb)
     cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'] , "filenames.jpg"), img_gray)
     cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'] , "filenames1.jpg"), thresholded_image)
     cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'] , "filenames11.jpg"), img_thresh)


     file1 = "filenames.jpg"
     file11 = "filenames1.jpg"
     file111 = "filenames11.jpg"



     counting = str(car_count) 
     return render_template('detection_result.html', filename=filename,file1=file1, file11=file11, file111=file111, files=file, count = counting)

@app.route('/display/<filename>')
def display_image(filename):
	print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='files/' + filename), code=301)

def make_grayscale(in_stream):
    # Credit: https://stackoverflow.com/a/34475270

    #use numpy to construct an array from the bytes
    arr = np.fromstring(in_stream, dtype='uint8')

    #decode the array into an image
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    # Make grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, out_stream = cv2.imencode('.PNG', gray)

    return out_stream


@app.route('/bantuan')
def bantuan():
     return render_template('bantuan.html')


if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)
