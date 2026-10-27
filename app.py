import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Estadísticas CNN",
    page_icon="📊",
    layout="wide"
)

st.title("Análisis de radiografías de tórax")
st.write(
    "Exploración del dataset y evaluación de un modelo "
    "CNN para clasificar imágenes NORMAL y PNEUMONIA."
)

# Resultados del entrenamiento realizado en Colab
epocas = [1, 2, 3, 4, 5]

accuracy_train = [0.8919, 0.9660, 0.9835, 0.9854, 0.9863]
accuracy_val = [0.9348, 0.9626, 0.9684, 0.9722, 0.9674]

loss_train = [0.2475, 0.0928, 0.0524, 0.0426, 0.0351]
loss_val = [0.1673, 0.1055, 0.0990, 0.0968, 0.1052]

# Métricas obtenidas sobre el conjunto de prueba
accuracy_test = 75.96
loss_test = 1.5002

# Matriz: filas = clase real, columnas = clase predicha
matriz = [[86, 148], [2, 388]]

st.header("1. Resultados del modelo")

c1, c2, c3 = st.columns(3)
c1.metric("Exactitud de prueba", f"{accuracy_test:.2f}%")
c2.metric("Pérdida de prueba", f"{loss_test:.4f}")
c3.metric("Épocas de entrenamiento", "5")

st.caption(
    "Resultados registrados durante la ejecución original en Colab. "
    "Las métricas de prueba corresponden a 624 imágenes."
)

st.header("2. Evolución del entrenamiento")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epocas, accuracy_train, marker="o", label="Entrenamiento")
    ax.plot(epocas, accuracy_val, marker="o", label="Validación")
    ax.set_title("Exactitud por época")
    ax.set_xlabel("Época")
    ax.set_ylabel("Exactitud")
    ax.set_xticks(epocas)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with col2:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epocas, loss_train, marker="o", label="Entrenamiento")
    ax.plot(epocas, loss_val, marker="o", label="Validación")
    ax.set_title("Pérdida por época")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_xticks(epocas)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

st.header("3. Matriz de confusión")

fig, ax = plt.subplots(figsize=(6, 4))
imagen = ax.imshow(matriz, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_xticklabels(["NORMAL", "PNEUMONIA"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["NORMAL", "PNEUMONIA"])
ax.set_xlabel("Clase predicha")
ax.set_ylabel("Clase real")
ax.set_title("Clasificación sobre el conjunto de prueba")

for i in range(2):
    for j in range(2):
        ax.text(
            j, i, str(matriz[i][j]),
            ha="center", va="center",
            color="black", fontsize=12
        )

fig.colorbar(imagen, ax=ax)
st.pyplot(fig)
plt.close(fig)

st.write(
    "El modelo clasificó correctamente 86 imágenes NORMAL y "
    "388 imágenes PNEUMONIA. Confundió 148 imágenes NORMAL con "
    "PNEUMONIA y 2 imágenes PNEUMONIA con NORMAL."
)

st.header("4. Métricas por clase")

metricas = pd.DataFrame({
    "Clase": ["NORMAL", "PNEUMONIA"],
    "Precisión": [0.98, 0.72],
    "Recall": [0.37, 0.99],
    "F1-score": [0.53, 0.84],
    "Imágenes evaluadas": [234, 390]
})

st.dataframe(
    metricas.style.format({
        "Precisión": "{:.2f}",
        "Recall": "{:.2f}",
        "F1-score": "{:.2f}"
    }),
    hide_index=True,
    use_container_width=True
)

st.caption(
    "Las métricas por clase están redondeadas según el reporte "
    "de clasificación generado en Colab."
)

st.header("5. Exploración del dataset")

archivo_csv = "tabla_limpia.csv"

if os.path.exists(archivo_csv):
    df = pd.read_csv(archivo_csv)

    c1, c2, c3 = st.columns(3)
    c1.metric("Imágenes en tabla limpia", f"{len(df):,}")
    c2.metric("Imágenes duplicadas restantes", f"{df.duplicated('huella').sum():,}")
    c3.metric("Formatos registrados", str(df["formato"].nunique()))

    st.subheader("Distribución por clase")
    conteo = df["tipo"].value_counts()
    st.bar_chart(conteo)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución del ancho")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["ancho"], bins=30)
        ax.set_xlabel("Ancho en píxeles")
        ax.set_ylabel("Número de imágenes")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Ancho y alto de las imágenes")
        fig, ax = plt.subplots(figsize=(6, 4))
        for clase in ["NORMAL", "PNEUMONIA"]:
            sub = df[df["tipo"] == clase]
            ax.scatter(
                sub["ancho"], sub["alto"],
                s=8, alpha=0.5, label=clase
            )
        ax.set_xlabel("Ancho en píxeles")
        ax.set_ylabel("Alto en píxeles")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Consultar los datos de la tabla limpia"):
        st.dataframe(df.head(100), use_container_width=True)

else:
    st.warning(
        "No se encontró tabla_limpia.csv. Sube ese archivo al "
        "repositorio de GitHub para mostrar las estadísticas del dataset."
    )

st.divider()
st.caption(
    "Proyecto académico de clasificación de radiografías mediante CNN. "
    "Los resultados son experimentales y no constituyen un diagnóstico médico."
)

