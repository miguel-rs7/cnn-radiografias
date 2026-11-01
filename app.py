import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay

st.set_page_config(page_title="Análisis de radiografías de tórax", page_icon="🩻", layout="wide")

# ---------- Resultados reales de UNA sola ejecución del notebook ----------
EPOCAS = [1, 2, 3, 4, 5]
ACC_TRAIN = [0.8622, 0.9569, 0.9724, 0.9835, 0.9832]
ACC_VAL = [0.9204, 0.9684, 0.9732, 0.9684, 0.9741]
LOSS_TRAIN = [0.3067, 0.1182, 0.0712, 0.0522, 0.0467]
LOSS_VAL = [0.1818, 0.1014, 0.0883, 0.1022, 0.0835]

ACC_TEST_CNN = 0.7404
LOSS_TEST_CNN = 1.3828
MATRIZ_CNN = np.array([[74, 160], [2, 388]])
REPORTE_CNN = pd.DataFrame({
    "Clase": ["NORMAL", "PNEUMONIA"], "Precisión": [0.97, 0.71],
    "Recall": [0.32, 0.99], "F1": [0.48, 0.83], "Imágenes": [234, 390]})

ACC_TEST_LOG = 0.7621
MATRIZ_LOG = np.array([[91, 140], [7, 380]])  # reconstruida desde el reporte de clasificación
REPORTE_LOG = pd.DataFrame({
    "Clase": ["NORMAL", "PNEUMONIA"], "Precisión": [0.93, 0.73],
    "Recall": [0.39, 0.98], "F1": [0.55, 0.84], "Imágenes": [231, 387]})

COMPARACION = pd.DataFrame([
    {"Modelo": "CNN", "Imágenes de prueba": 624, "Exactitud": 0.7404,
     "Precisión neumonía": 0.7080, "Recall neumonía": 0.9949,
     "Recall normal": 0.32, "F1 neumonía": 0.8273},
    {"Modelo": "Regresión logística", "Imágenes de prueba": 618, "Exactitud": 0.7621,
     "Precisión neumonía": 0.7308, "Recall neumonía": 0.9819,
     "Recall normal": 0.39, "F1 neumonía": 0.8379},
])


def dibujar_matriz(matriz):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ConfusionMatrixDisplay(confusion_matrix=matriz,
                           display_labels=["NORMAL", "PNEUMONIA"]).plot(ax=ax, values_format="d", cmap="Blues")
    st.pyplot(fig)
    plt.close(fig)


st.title("🩻 Análisis de radiografías de tórax")
st.write("Clasificación de radiografías NORMAL vs PNEUMONIA con dos modelos: una red neuronal "
         "convolucional (CNN) y una regresión logística. Dataset: Chest X-Ray Images (Pneumonia), Kaggle.")
st.warning("Proyecto académico. No es una herramienta de diagnóstico médico.")

tab_datos, tab_cnn, tab_log, tab_comp, tab_clas = st.tabs(
    ["📁 Datos", "🧠 Modelo 1: CNN", "📈 Modelo 2: Regresión logística",
     "⚖️ Comparación", "🔍 Clasificar radiografía"])


@st.cache_data
def cargar_tabla():
    if os.path.exists("tabla_limpia.csv"):
        return pd.read_csv("tabla_limpia.csv")
    return None


# ---------------- Pestaña 1: datos ----------------
with tab_datos:
    st.header("Información del dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Imágenes originales", "5.856")
    c2.metric("Duplicadas eliminadas", "32")
    c3.metric("Imágenes tras limpieza", "5.824")
    c4.metric("Clases", "2")
    st.caption("Partición original de Kaggle: 5.216 en entrenamiento, 624 en prueba y 16 en validación. "
               "Los duplicados se detectaron con una huella MD5 de cada archivo.")

    df = cargar_tabla()
    if df is None:
        st.info("No se encontró tabla_limpia.csv en el repositorio. Súbelo junto a app.py "
                "para ver la distribución de clases y las estadísticas de tamaño.")
    else:
        st.subheader("Distribución de clases")
        conteo = df["tipo"].value_counts()
        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            barras = ax.bar(conteo.index, conteo.values, color=["#4C72B0", "#DD8452"])
            ax.bar_label(barras)
            ax.set_ylabel("Imágenes")
            st.pyplot(fig)
            plt.close(fig)
        with col_b:
            tabla = pd.crosstab(df["grupo"], df["tipo"])
            tabla["Total"] = tabla.sum(axis=1)
            st.dataframe(tabla)
            pct = conteo["PNEUMONIA"] / conteo.sum() * 100
            st.write(f"**{pct:.1f} %** de las imágenes son de neumonía. "
                     "Las clases están desbalanceadas, y eso afecta a los modelos.")

        st.subheader("Tamaño y peso de las imágenes")
        st.dataframe(df[["ancho", "alto", "peso_kb"]].describe().round(1))

        st.subheader("Ancho vs alto por clase")
        fig, ax = plt.subplots(figsize=(6, 4))
        for tipo, color in [("NORMAL", "#4C72B0"), ("PNEUMONIA", "#DD8452")]:
            sub = df[df["tipo"] == tipo]
            ax.scatter(sub["ancho"], sub["alto"], s=5, label=tipo, color=color)
        ax.set_xlabel("Ancho (px)")
        ax.set_ylabel("Alto (px)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Las imágenes más grandes son casi todas NORMAL. Esto sugiere que el tamaño "
                   "del archivo podría dar pistas que no tienen que ver con los pulmones.")

        st.subheader("Correlación entre variables del archivo")
        corr = df[["ancho", "alto", "peso_kb"]].corr()
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        im = ax.imshow(corr, cmap="YlGnBu")
        ax.set_xticks(range(3), corr.columns)
        ax.set_yticks(range(3), corr.columns)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
        fig.colorbar(im)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Solo mide la relación entre ancho, alto y peso del archivo. "
                   "No dice nada sobre el diagnóstico.")

# ---------------- Pestaña 2: modelo 1, CNN ----------------
with tab_cnn:
    st.header("Modelo 1: red neuronal convolucional (CNN)")
    st.write("Arquitectura: 2 capas convolucionales (32 y 64 filtros) con MaxPooling, una capa densa "
             "de 64 neuronas y salida sigmoide. Imágenes a color de 150×150, 5 épocas, optimizador Adam. "
             "Entrenamiento: 4.173 imágenes. Validación: 1.043. Prueba: 624.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Exactitud en validación (época 5)", f"{ACC_VAL[-1]:.2%}")
    m2.metric("Exactitud en prueba", f"{ACC_TEST_CNN:.2%}")
    m3.metric("Pérdida en prueba", f"{LOSS_TEST_CNN:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(EPOCAS, ACC_TRAIN, marker="o", label="Entrenamiento")
        ax.plot(EPOCAS, ACC_VAL, marker="o", label="Validación")
        ax.set_xlabel("Época"); ax.set_ylabel("Exactitud"); ax.set_xticks(EPOCAS)
        ax.set_title("Exactitud"); ax.legend(); ax.grid(True)
        st.pyplot(fig); plt.close(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(EPOCAS, LOSS_TRAIN, marker="o", label="Entrenamiento")
        ax.plot(EPOCAS, LOSS_VAL, marker="o", label="Validación")
        ax.set_xlabel("Época"); ax.set_ylabel("Pérdida"); ax.set_xticks(EPOCAS)
        ax.set_title("Pérdida"); ax.legend(); ax.grid(True)
        st.pyplot(fig); plt.close(fig)

    st.subheader("Matriz de confusión (conjunto de prueba)")
    dibujar_matriz(MATRIZ_CNN)
    st.subheader("Reporte por clase")
    st.dataframe(REPORTE_CNN, hide_index=True)

    st.info("La CNN detecta 388 de 390 neumonías (recall 99,5 %), pero solo reconoce 74 de 234 "
            "radiografías normales (recall 31,6 %): marca como enfermas a 160 personas sanas. "
            "Además, pasa de 97 % en validación a 74 % en prueba, señal de sobreajuste o de que "
            "el conjunto de prueba es distinto al de entrenamiento.")

# ---------------- Pestaña 3: modelo 2, regresión logística ----------------
with tab_log:
    st.header("Modelo 2: regresión logística")
    st.write("Cada radiografía se convierte a gris, se reduce a 64×64 píxeles y se aplana en 4.096 "
             "variables (valores entre 0 y 1). Se entrena con pesos de clase balanceados "
             "(`class_weight='balanced'`) y hasta 1.000 iteraciones. "
             "Entrenamiento: 5.190 imágenes. Prueba: 618 (sin duplicados).")
    st.metric("Exactitud en prueba", f"{ACC_TEST_LOG:.2%}")

    st.subheader("Matriz de confusión (conjunto de prueba)")
    dibujar_matriz(MATRIZ_LOG)
    st.caption("Matriz reconstruida a partir de las métricas del reporte de clasificación.")
    st.subheader("Reporte por clase")
    st.dataframe(REPORTE_LOG, hide_index=True)

    st.info("La regresión logística detecta 380 de 387 neumonías (recall 98 %), pero solo reconoce "
            "91 de 231 radiografías normales (recall 39 %). Aun siendo un modelo mucho más simple, "
            "su exactitud (76,21 %) es similar a la de la CNN.")

# ---------------- Pestaña 4: comparación ----------------
with tab_comp:
    st.header("CNN vs regresión logística")
    st.dataframe(
        COMPARACION.style.format({
            "Exactitud": "{:.2%}", "Precisión neumonía": "{:.2%}",
            "Recall neumonía": "{:.2%}", "Recall normal": "{:.0%}", "F1 neumonía": "{:.2%}"}),
        hide_index=True)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(COMPARACION))
    w = 0.35
    ax.bar(x - w / 2, COMPARACION["Recall neumonía"], w, label="Recall neumonía")
    ax.bar(x + w / 2, COMPARACION["Recall normal"], w, label="Recall normal")
    ax.set_xticks(x, COMPARACION["Modelo"])
    ax.set_ylim(0, 1.05); ax.legend(); ax.set_title("Qué tan bien detecta cada clase")
    st.pyplot(fig); plt.close(fig)

    st.info("Los dos modelos quedan alrededor del 75 % de exactitud. La regresión logística (76,21 %) "
            "no es peor que la CNN (74,04 %). Ambos detectan casi toda la neumonía, pero fallan "
            "con las radiografías normales.")
    st.caption("Nota: la logística se evaluó con 618 imágenes (tras quitar duplicados) y la CNN con "
               "las 624 originales, así que la comparación es aproximada. La CNN no usa pesos de clase.")


# ---------------- Pestaña 5: clasificar con la CNN ----------------
@st.cache_resource
def cargar_cnn():
    import tensorflow as tf
    return tf.keras.models.load_model("modelo_cnn.keras")


with tab_clas:
    st.header("Clasificar una radiografía")
    st.write("Sube una radiografía de tórax (jpg, jpeg o png). La CNN devuelve la probabilidad de neumonía.")
    archivo = st.file_uploader("Radiografía", type=["jpg", "jpeg", "png"])

    if archivo is not None:
        imagen = Image.open(archivo).convert("RGB")
        st.image(imagen, caption="Radiografía cargada", width=350)

        # El modelo ya incluye Rescaling(1/255): NO se divide aquí.
        entrada = np.expand_dims(np.array(imagen.resize((150, 150)), dtype=np.float32), axis=0)

        with st.spinner("Analizando..."):
            prob = float(cargar_cnn().predict(entrada, verbose=0)[0][0])

        if prob >= 0.5:
            st.error(f"Resultado: **PNEUMONIA** (probabilidad {prob:.1%})")
        else:
            st.success(f"Resultado: **NORMAL** (probabilidad de neumonía {prob:.1%})")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption("Recuerda: en las pruebas, la CNN marcó como neumonía a 160 de 234 radiografías "
                   "normales. Un resultado de neumonía puede ser una falsa alarma. "
                   "Esto no reemplaza la valoración de un médico.")
