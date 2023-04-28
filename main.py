from flask import Flask  , request
import json
import numpy as np
from keras import layers
from keras import models
from keras import optimizers
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from io import BytesIO
from PIL import Image
import base64
app = Flask(__name__)
map_label = {
    0 : "Ca Chep",
    1 : "Chan trau thoi sao",
    2 : "Dam cuoi chuot",
    3 : "Dan ga me con",
    4 : "Dan lon am duong",
    5 : "Danh ghen",
    6 : "Ga Da Xuong",
    7 : "Ga KH",
    8 : "Hai ba trung",
    9 : "Hai dua",
    10 : "Hoi lang",
    11 : "Lon KH",
    12 : "Ly ngu vong nguyet",
    13 :"Ngo quyen",
    14 :"Ngu ho",
    15 :"Phat quan am",
    16 :"Thai bach kim tinh",
    17 :"Thanh mau",
    18 :"Thay do coc",
    19 :"To nu",
    20 :"Tu phu",
    21 :"Vinh hoa phu quy",
    22 :"Vinh quy bai to",
    23 :"dasd"
}

@app.route('/predict_label', methods=['POST'])
def predict_label():
    img = request.form.get('image')
    print(img)
    # return img
    img_list = [read_base64(img)]
    test_batch = np.stack([preprocess_input(np.array(img.resize((240,240)))) for img in img_list])
    test_batch /= 255.0
    model_cdcn = models.Sequential()
    model_cdcn.add(conv_base)
    model_cdcn.add(layers.Flatten())
    model_cdcn.add(layers.Dense(512, activation='relu'))
    model_cdcn.add(layers.Dropout(0.5))
    model_cdcn.add(layers.Dense(128, activation='relu'))
    model_cdcn.add(layers.Dropout(0.3))
    model_cdcn.add(layers.Dense(23, activation='softmax'))
    model_cdcn.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=2e-5),
                  metrics=['acc'])
    model_cdcn.load_weights(MODEL_FILE)
    pred_probs = model_cdcn.predict(test_batch)
    pred_probs.tolist()
    max_value = max(pred_probs[0])
    max_index = pred_probs[0].argmax()
    return json.dumps({
        "index_img":str(max_index),
        "accuracy" : str(max_value)
    })


def read_base64(base64_img):
    decoded_img = base64.b64decode(base64_img.split(',')[1])
    img = Image.open(BytesIO(decoded_img))
    return img

if __name__ == '__main__':
    MODEL_FILE = './models/model_gen_img_tranh_vgg_v1.h5'
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(240, 240, 3))
    app.run(host='0.0.0.0', port=12345, debug=False)
