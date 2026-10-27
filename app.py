import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Clasificación de radiografías")
st.write("Detección de neumonía mediante CNN")

@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelo_cnn.keras")

archivo = st.file_uploader(
    "Sube una radiografía de tórax",
    type=["jpg", "jpeg", "png"]
)

if archivo is not None:
    imagen = Image.open(archivo).convert("RGB")
    st.image(imagen, caption="Radiografía cargada")

    imagen = imagen.resize((150, 150))
    imagen = np.array(imagen, dtype=np.float32)
    imagen = np.expand_dims(imagen, axis=0)

    modelo = cargar_modelo()
    prediccion = modelo.predict(imagen, verbose=0)[0][0]

    if prediccion >= 0.5:
        st.write("Resultado: PNEUMONIA")
    else:
        st.write("Resultado: NORMAL")
