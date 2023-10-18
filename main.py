import streamlit as st
from PIL import Image
from keras.models import *
import cv2
import numpy as np
from tempfile import NamedTemporaryFile

loaded_model = load_model("./model/COVID_predictor_model.h5")
image_file = st.file_uploader("Upload an X-ray file")


def load_image(image_file):
    img = Image.open(image_file)
    return img


def process_image(image_f):
    image_f = image_f/255
    image_f = cv2.resize(image_f, (224,224))
    return image_f


def predict(image_path, model):
    im = cv2.imread(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    processed_test_image = np.expand_dims(processed_test_image, axis=0)

    ps = loaded_model.predict(processed_test_image)
    if round(ps[0][0], 1) == 0:
        msg = {"Report": "COVID detected with a percentage of {}%".format(round((1-ps[0][0])*100, 2))}
    else:
        msg = {"Report": "Normal"}
    return msg


# To See details
if image_file:
    buffer = image_file
    temp_file = NamedTemporaryFile(delete=False)

    # To View Uploaded Image
    if buffer:
        temp_file.write(buffer.getvalue())
        st.image(load_image(image_file),width=250)
    st.write("Prediction results")
    st.write(predict(temp_file.name, loaded_model))
