from flask import Flask, render_template, request
from collections import OrderedDict
import json
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


app = Flask(__name__)
max_size = 10 * 1024 * 1024  # LIMIT : MAX UPLOAD SIZE TO 10 MB
breed_list = [
    'scottish_deerhound',
    'maltese_dog',
    'afghan_hound',
    'entlebucher',
    'bernese_mountain_dog'
]
model = None


def getModel():
    loaded_model = load_model('./static/model/resnet_model.h5')
    print('Model Loading finished.')
    return loaded_model


model = getModel()


def preprocessImage(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, 0)
    return image


@app.route('/')
def index():
    return render_template('Prediction/index.html')


@app.route('/about')
def about():
    return render_template('Prediction/about.html', breed_list=breed_list, length=int(len(breed_list) / 3))


supported_type = ['image/png', 'image/jpeg']


@app.route('/prediction', methods=['POST'])
def prediction():
    if 'file' not in request.files:
        # No file
        return {"Lỗi": "Không có tệp nào được tìm thấy."}, 400

    if request.content_length > (max_size + 1000):
        return {"Lỗi": f"Kích thước tệp phải nhỏ hơn {max_size / (1024 * 1024)} MB"}, 400

    file = request.files['file']

    # if user does not select file, browser also submit an empty part without filename
    if file.filename == '':
        return {"Lỗi": "Không có tệp tin nào được chọn."}, 400

    mime_type = file.content_type
    if file and mime_type in supported_type:
        image = Image.open(file)
        processed_image = preprocessImage(image, target_size=(224, 224))
        global model
        if model is None:
            model = getModel()

        predicted_data = model.predict(processed_image)
        sorted_index = predicted_data.argsort()
        top_5_prediction = OrderedDict()
        for i in range(1, 6):
            top_5_prediction[breed_list[sorted_index[0, -i]]] = str(predicted_data[0, sorted_index[0, -i]])

        # success return
        return json.dumps(top_5_prediction), 200
    else:
        return {"Lỗi": "Định dạng tệp không được hỗ trợ."}, 400


@app.errorhandler(404)
def exception(ex):
    return render_template('404.html')


if __name__ == '__main__':
    print('Loading Model...')
    app.run()
