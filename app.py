import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

st.set_page_config(
    page_title="Análisis de radiografías",
    page_icon="🩻",
    layout="wide"
)

st.title("Análisis de radiografías de tórax")
st.write(
    "Clasificación de radiografías mediante una red neuronal "
    "convolucional (CNN) y comparación con regresión logística."
)

# Resultados obtenidos durante la evaluación
resultados = pd.DataFrame([
    {
        "Modelo": "CNN",
        "Exactitud": 0.7596,
        "Precisión neumonía": 0.7240,
        "Recall neumonía": 0.9949,
        "F1 neumonía": 0.8380
    },
    {
        "Modelo": "Regresión logística",
        "Exactitud": 0.7621,
        "Precisión neumonía": 0.7308,
        "Recall neumonía": 0.9819,
        "F1 neumonía": 0.8379
    }
])

tab1, tab2, tab3 = st.tabs([
    "Estadísticas CNN",
    "Comparación de modelos",
    "Clasificar radiografía"
])

# Pestaña 1: estadísticas CNN
with tab1:
    st.header("Evaluación del modelo CNN")

    col1, col2 = st.columns(2)
    col1.metric("Exactitud en prueba", "75,96 %")
    col2.metric("Pérdida en prueba", "1,5002")

    st.subheader("Exactitud durante el entrenamiento")

    epocas = [1, 2, 3, 4, 5]
    accuracy_train = [0.8919, 0.9660, 0.9835, 0.9854, 0.9863]
    accuracy_val = [0.9348, 0.9626, 0.9684, 0.9722, 0.9674]

    fig, ax = plt.subplots()
    ax.plot(epocas, accuracy_train, marker="o", label="Entrenamiento")
    ax.plot(epocas, accuracy_val, marker="o", label="Validación")
    ax.set_xlabel("Época")
    ax.set_ylabel("Exactitud")
    ax.set_xticks(epocas)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Pérdida durante el entrenamiento")

    loss_train = [0.2475, 0.0928, 0.0524, 0.0426, 0.0351]
    loss_val = [0.1673, 0.1055, 0.0990, 0.0968, 0.1052]

    fig, ax = plt.subplots()
    ax.plot(epocas, loss_train, marker="o", label="Entrenamiento")
    ax.plot(epocas, loss_val, marker="o", label="Validación")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida")
    ax.set_xticks(epocas)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Matriz de confusión de la CNN")

    matriz_cnn = np.array([[86, 148], [2, 388]])

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix=matriz_cnn,
        display_labels=["NORMAL", "PNEUMONIA"]
    ).plot(ax=ax, values_format="d")
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "La matriz muestra los resultados registrados en la evaluación "
        "del conjunto de prueba."
    )

# Pestaña 2: comparación
with tab2:
    st.header("Comparación de modelos")

    st.dataframe(
        resultados.style.format({
            "Exactitud": "{:.2%}",
            "Precisión neumonía": "{:.2%}",
            "Recall neumonía": "{:.2%}",
            "F1 neumonía": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Exactitud por modelo")
    fig, ax = plt.subplots()
    ax.bar(resultados["Modelo"], resultados["Exactitud"])
    ax.set_ylabel("Exactitud")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=10)
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "La regresión logística obtuvo una exactitud ligeramente superior. "
        "La CNN presentó un recall de neumonía más alto. "
        "Estos resultados corresponden a la evaluación realizada y no "
        "demuestran por sí solos que un modelo sea clínicamente adecuado."
    )

# Pestaña 3: predicción de una imagen
with tab3:
    st.header("Clasificar una radiografía")
    st.warning(
        "Esta demostración es educativa y no debe utilizarse para "
        "diagnosticar ni descartar neumonía."
    )

    archivo = st.file_uploader(
        "Sube una radiografía de tórax",
        type=["jpg", "jpeg", "png"]
    )

    if archivo is not None:
        try:
            imagen_original = Image.open(archivo).convert("RGB")
            st.image(
                imagen_original,
                caption="Radiografía cargada",
                use_container_width=True
            )

            imagen = imagen_original.resize((150, 150))
            imagen = np.array(imagen, dtype=np.float32)
            imagen = np.expand_dims(imagen, axis=0)

            @st.cache_resource
            def cargar_modelo():
                return tf.keras.models.load_model("modelo_cnn.keras")

            modelo = cargar_modelo()
            prediccion = float(
                modelo.predict(imagen, verbose=0)[0][0]
            )

            st.write(f"Salida del modelo: {prediccion:.4f}")

            if prediccion >= 0.5:
                st.subheader("Resultado del modelo: PNEUMONIA")
            else:
                st.subheader("Resultado del modelo: NORMAL")

            st.caption(
                "La salida es una predicción del modelo, no una probabilidad "
                "clínicamente validada."
            )

        except Exception as error:
            st.error(f"No se pudo procesar la imagen: {error}")
