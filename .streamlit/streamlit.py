import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss

# Función para plotear el histograma y el ajuste
def plot_histogram(dataset, m, hist_color, fit_color, mean_color, std_dev_color):
    # Obtener los datos
    if dataset == "todos":
        data = pd.concat(listas.values())[:m]
    else:
        data = listas[dataset][:m]

    # Crear histograma
    plt.hist(data, bins=np.arange(min(data), max(data) + 1) - 0.5, density=True, alpha=0.6, color=hist_color,
             edgecolor='black', linewidth=1.2, label='Datos experimentales')

    # Ajuste de la distribución binomial
    fitted_results = ss.fit(ss.binom, data, bounds=[(0, 100), (0, 1)])
    p, n = fitted_results[0]
    x = np.arange(0, max(data) + 1)
    y = ss.binom.pmf(x, n, p)

    # Para el ajuste
    plt.plot(x, y, 'r--', linewidth=1.5,
             label=f'Ajuste Binomial\nProbabilidad de éxito (p): {p:.2f}\nNúmero de ensayos (n): {n:.2f}')

    plt.xlabel('Número de Caras')
    plt.ylabel('Densidad de probabilidad')
    plt.title(f'Histograma y Ajuste Binomial para los primeros {m} tiros del conjunto de datos "{dataset}"')
    plt.legend()
    plt.grid(True)

    # Ajustar la posición del cuadro de texto
    plt.tight_layout()

    st.pyplot()

# Crear la interfaz de usuario
def main():
    st.title('Ajuste Binomial y Histograma Interactivo')
    dataset = st.selectbox('Selecciona un conjunto de datos:', ["Clase"] + list(listas.keys()))

    if dataset == "Clase":
        dataset_selected = "todos"
    else:
        dataset_selected = dataset

    m = st.slider('Selecciona el valor de m:', 1, 500, 100)
    hist_color = st.color_picker('Color del histograma:', '#00f')
    fit_color = st.color_picker('Color del ajuste:', '#f00')
    mean_color = st.color_picker('Color del valor mínimo:', '#0f0')
    std_dev_color = st.color_picker('Color de la desviación estándar:', '#ffa500')

    plot_histogram(dataset_selected, m, hist_color, fit_color, mean_color, std_dev_color)


if __name__ == '__main__':
    main()

