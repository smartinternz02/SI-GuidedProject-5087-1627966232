
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model = load_model("Image.h5")


@app.route('/')
def index():
    return render_template('animalClassification.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath, 'upload', f.filename)
        print("This animal is", filepath)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(64, 64))
        x = image.img_to_array(img)
        print(x)
        x = np.expand_dims(x, axis=0)
        print(x)
        preds = model.predict(x)
        preds = np.argmax(preds, axis=1)
        print("prediction", preds)
        index = ["cat", "eagle", "eel", "elephant", "elk", "fish", "hen", "iguana", "lemur", "newt"]
        text = "This Animal is a/an : " + str(index[preds[0]])
    return text


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
