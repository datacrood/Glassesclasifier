import streamlit as st
import requests
from PIL import Image
import io

st.title("Face with Glasses or Without Glasses Classifier")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    resized_image =  image.resize((250,250))
    st.image(resized_image, caption='Uploaded Image.')

    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()

    response = requests.post(
        "https://glassornoglassclassifierapi.onrender.com/predict",
        files={"file": ("image.jpg", img_bytes, "image/jpeg")}
    )

    if response.status_code == 200:
        prediction = response.json()['prediction']
        st.write(f"Prediction: {prediction}")
    else:
        st.write("Error: ", response.json().get('error'))
