# Step-1 Create python environment

# Step-2 Activate the environment. 
# 
# Step-3 Upgrade the installer, for e.g., "pip install --upgrade pip", and then install streamlit and tensorflow inside it. 
# Follow this link: https://www.activestate.com/resources/quick-reads/how-to-install-keras-and-tensorflow/

# Step-4 streamlit run <filename>
# Step-5 If it runs properly, come back to the terminal and create a requirements file with pip3 freeze requirements.txt
# Step-6 create a github repo and upload your code file, model.h5 file, labels.txt, and requirements.txt file in the repo.
# Step-7 Go to streamlit, create an account, upload your repo on streamlit and let it host. 

from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st 


def classify_animal(img):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("./2classes.h5", compile=False)

    # Load the labels
    class_names = open("./labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Replace this with the path to your image
    image = img.convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]



    return class_name, confidence_score




st.set_page_config(layout='wide')

st.title("Animal Classify App")

input_img = st.file_uploader("Enter your image", type=['jpg', 'png', 'jpeg'])

if input_img is not None:
    if st.button("Classify"):
        
        col1, col2, col3 = st.columns([1,1,1])

        with col1:
            st.info("Your uploaded Image")
            st.image(input_img, use_column_width=True)

        with col2:
            st.info("Your Result")
            image_file = Image.open(input_img)
            label, confidence_score = classify_animal(image_file)
            st.success(label)
          