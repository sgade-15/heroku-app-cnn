from flask import Flask, render_template, request
import numpy
import cv2
import os

app = Flask(__name__)

from Inference import get_plant_disease, background_removal

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        if 'file' not in request.files:
            print("files not uploaded")
            return
        #file = request.files['file']
        #read image file string data
        filestr = request.files['file'].read()
        #convert string data to numpy array
        npimg = numpy.fromstring(filestr, numpy.uint8)
        # convert numpy array to image
        img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
        dim = (256, 256)
        # resize image
        image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
        foreground_image = background_removal(image_bytes=image)
        top1_prob, disease_name, top3_disease, top3_prob = get_plant_disease(image_bytes=foreground_image)
        return render_template('result.html', disease= disease_name, probability=top1_prob, top3= top3_disease, top3_prob= top3_prob)


if __name__ == '__main__':
    app.run(debug=True,port=os.getenv('PORT',5000))
