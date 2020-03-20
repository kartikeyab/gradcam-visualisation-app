from tensorflow.keras.applications import VGG16
from pyimagesearch.gradcam import GradCAM
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from keras.models import load_model
from matplotlib import pyplot
import imutils
import cv2
import streamlit as st
import os
import numpy as np
from PIL import Image


st.title("VGG GradCam Visualisations")

st.write("This is a webapp which let's you visualise the Gradient Weighted Class Activation Mappings (Grad-Cam) of a \n"
        "pre-trained VGG16 Convolutional Neural Net, trained on the ImageNet dataset. You also have the\n"
        "flexibility to select a specific layer of the model which will change the mappings accordingly. \n"
        "The output of the model is a heatmap which  highlights the pixels in the image which are \n"
        "picked up by the model to classify the given image into a certain class.")


def preprocess(img, req_size = (224,224)):
    image = Image.fromarray(img.astype('uint8'))
    image = image.resize(req_size)
    face_array = img_to_array(image)
    face_array = np.expand_dims(face_array, 0)
    return face_array

#Face image upload
upload_image = st.file_uploader("Upload Image")
if upload_image is not None:
    file_bytes = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    st.image(opencv_image, width = 270, channels="BGR",use_column_width=True)
    opencv_image_processed = preprocess(opencv_image)

#load model
model = VGG16(weights = "imagenet")

#Layer output
layer_name = st.multiselect(
    "Choose the layer",
    ('block5_pool', 'block5_conv3', 'block5_conv2', 'block5_conv1', 'block4_pool', 'block4_conv3',
    'block4_conv2', 'block4_conv1', 'block3_pool'))


if st.button('Visualise'):
    preds = model.predict(opencv_image_processed)
    i = np.argmax(preds[0])
    decoded = imagenet_utils.decode_predictions(preds)
    (imagenetID, label, prob) = decoded[0][0]
    label = "{}: {:.2f}%".format(label, prob * 100)
    print("[INFO] {}".format(label))

    # initialize our gradient class activation map and build the heatmap
    cam = GradCAM(model, i, str(layer_name[0]))             #layer parameter
    heatmap = cam.compute_heatmap(opencv_image_processed)

    # resize the resulting heatmap to the original input image dimensions
    # and then overlay heatmap on top of the image
    heatmap = cv2.resize(heatmap, (opencv_image.shape[1], opencv_image.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, opencv_image, alpha=0.5)

    # draw the predicted label on the output image
    cv2.rectangle(output, (0, 0), (340,30), (0, 0, 0), -1)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2)

    # display the original image and resulting heatmap and output image
    # to our screen
    output = np.hstack([heatmap, output])
    output = imutils.resize(output, height=2700)
    st.image(output, channels="BGR",use_column_width=True)
    
    #cv2.imshow("Output", output)
    #pyplot.imsave('output.jpg', output)
    #cv2.waitKey(0)