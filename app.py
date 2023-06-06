import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pickle
import matplotlib.pyplot as plt
# import tensorflow as tf
# tf.keras.backend.set_image_data_format('channels_first')

st.set_page_config(layout='wide')

# @st.cache_resource
# def load_random_forest_model():
#     return pickle.load(open("random_forest.pickle", "rb"))

@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model('mnist_cnn.h5')

# model_rf = load_random_forest_model()
model_cnn = load_cnn_model()

chosen_model = st.sidebar.selectbox(
#     "Model:", ("Random Forest", "Convolutional Network")
    "Model:", ("Convolutional Network")
)
# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 10)
realtime_update = st.sidebar.checkbox("Update in realtime", True)


col1, col2,col3= st.columns([4, 4, 3])

with col1:
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color='#FFFFFF',
        background_color='#000000',
        update_streamlit=realtime_update,
        height=280,
        width=280,
        drawing_mode=drawing_mode,
        key="canvas",
    )

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    col2.image(canvas_result.image_data, width=280)
# st.image(canvas_result.image_data[:,:,0], width=280)
# =======================================================================

smaller_img = canvas_result.image_data[::10, ::10,0]
# st.text(smaller_img.flatten().shape)
if chosen_model == "Random Forest":
    model = model_rf
    img_pred = smaller_img.reshape(1, -1)
else:
    smaller_img = smaller_img.astype('float32')
    smaller_img /= 255
    img_pred = smaller_img.reshape(1, 28, 28)
    model = model_cnn
# model = pickle.load(open("random_forest.pickle", "rb"))

# st.text(smaller_img.reshape(1, -1).shape)
# st.text(smaller_img.reshape(1, -1))
col3.subheader("Predição:")
if chosen_model == "Random Forest":
    col3.subheader(model.predict(img_pred)[0])
else:
    yhat = model.predict(img_pred)
    col3.subheader(yhat.argmax(axis=1)[0])

col3.caption("O que o modelo vê:")
col3.image(smaller_img)
