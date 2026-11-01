import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Análisis de radiografías de tórax")
st.write("Dataset: Chest X-Ray Images (Pneumonia), Kaggle.")

tab_datos, tab_clasificar = st.tabs(["Datos", "Clasificar radiografía"])


@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelo_cnn.keras")


with tab_datos:
    st.header("Información del dataset")

    c1, c2, c3 = st.columns(3)
    c1.metric("Imágenes originales", "5.856")
    c2.metric("Duplicadas eliminadas", "32")
    c3.metric("Imágenes tras limpieza", "5.824")

    st.write(
        "Partición original de Kaggle: 5.216 imágenes de entrenamiento, "
        "624 de prueba y 16 de validación."
    )
    st.write(
        "Clases: NORMAL y PNEUMONIA. Hay unas 3 radiografías de neumonía "
        "por cada una normal, es decir, las clases están desbalanceadas."
    )
    st.write(
        "Tamaño medio de las imágenes: 1.327 px de ancho por 970 px de alto, "
        "con un peso medio de 206 KB. Los duplicados se detectaron con una "
        "huella MD5 de cada archivo."
    )


with tab_clasificar:
    st.header("Clasificar una radiografía")

    archivo = st.file_uploader(
        "Sube una radiografía de tórax",
        type=["jpg", "jpeg", "png"]
    )

    if archivo is not None:
        imagen = Image.open(archivo).convert("RGB")
        st.image(imagen, caption="Radiografía cargada", width=350)

        imagen = imagen.resize((150, 150))
        imagen = np.array(imagen, dtype=np.float32)
        imagen = np.expand_dims(imagen, axis=0)

        modelo = cargar_modelo()
        prediccion = float(modelo.predict(imagen, verbose=0)[0][0])

        st.write(f"Salida del modelo: {prediccion:.4f}")

        if prediccion >= 0.5:
            st.write("Resultado: PNEUMONIA")
        else:
            st.write("Resultado: NORMAL")
