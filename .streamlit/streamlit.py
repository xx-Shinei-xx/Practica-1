import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

st.set_option('deprecation.showPyplotGlobalUse', False)



# Definir los datos
listas = {
    Guillermo y Shawn": [2, 4, 5, 3, 7, 3, 4, 6, 4, 4, 3, 5, 3, 2, 3, 4, 8, 6, 4, 2, 5, 5, 3, 8, 4, 7, 4, 6, 3, 5, 8, 7, 3, 3, 6, 5, 4, 4, 5, 2, 5, 3, 7, 6, 3, 6, 5, 2, 4, 6, 5, 4, 6, 3, 6, 5, 3, 7, 8, 7, 4, 4, 4, 8, 5, 4, 3, 5, 7, 5, 2, 2, 3, 5, 1, 6, 4, 6, 4, 4, 3, 3, 6, 6, 3, 4, 5, 5, 5, 7, 6, 7, 4, 3, 5, 4, 5, 7, 6],
    "Lobsang y Rebeca": [6, 5, 5, 6, 5, 4, 6, 6, 5, 4, 6, 6, 5, 6, 9, 8, 1, 7, 5, 3, 5, 3, 3, 4, 3, 5, 4, 4, 6, 2, 5, 6, 7, 5, 5, 2, 3, 5, 7, 6, 5, 1, 6, 4, 4, 8, 5, 3, 6, 5, 6, 4, 5, 5, 3, 2, 6, 5, 2, 9, 7, 4, 7, 4, 3, 3, 6, 6, 4, 4, 6, 5, 5, 4, 6, 4, 9, 6, 4, 4, 8, 6, 4, 4, 8, 6, 8, 3, 6, 2, 5, 6, 2, 4, 5, 3, 4, 6, 5, 7],
    "Diego y Saul": [5, 6, 5, 5, 4, 4, 4, 3, 6, 5, 4, 7, 5, 7, 3, 5, 4, 7, 3, 4, 6, 3, 4, 5, 6, 2, 7, 3, 6, 2, 4, 7, 5, 5, 5, 3, 6, 6, 5, 4, 4, 7, 4, 7, 6, 5, 4, 4, 3, 5, 5, 4, 4, 7, 4, 5, 5, 4, 7, 6, 9, 5, 5, 5, 4, 5, 5, 7, 5, 4, 8, 3, 4, 4, 4, 8, 4, 9, 7, 7, 5, 5, 7, 5, 4, 4, 6, 7, 4, 2, 5, 5, 3, 6, 7, 5, 4, 4, 4, 7],
    "Giovanna y Mario": [5, 5, 5, 7, 7, 4, 7, 4, 6, 6, 6, 5, 5, 5, 6, 6, 6, 6, 7, 4, 5, 4, 5, 6, 5, 8, 7, 4, 3, 6, 4, 3, 6, 2, 7, 5, 8, 7, 6, 7, 4, 5, 5, 6, 4, 7, 4, 6, 4, 3, 4, 5, 5, 4, 3, 5, 6, 7, 5, 4, 5, 4, 4, 4, 6, 8, 6, 7, 5, 1, 3, 6, 4, 5, 4, 3, 5, 4, 3, 4, 6, 8, 5, 6, 5, 7, 5, 4, 6, 5, 4, 4, 10, 8, 3, 7, 5, 5, 4, 4],
    "Dessiré y Fabricio": [6, 4, 3, 6, 6, 6, 6, 7, 4, 4, 5, 4, 5, 3, 4, 8, 5, 3, 6, 6, 6, 5, 5, 5, 6, 4, 6, 6, 7, 6, 6, 6, 5, 4, 2, 5, 3, 6, 4, 4, 6, 5, 3, 4, 5, 5, 6, 5, 7, 5, 3, 3, 5, 5, 5, 4, 10, 5, 6, 4, 3, 5, 6, 4, 3, 4, 6, 5, 4, 6, 8, 5, 5, 5, 4, 5, 8, 4, 5, 5, 3, 3, 4, 5, 2, 7, 4, 5, 4, 6, 5, 6, 3, 6, 5, 7, 7, 9, 5, 3],
    "Jacobo y Cesar": [6, 6, 5, 3, 2, 5, 7, 8, 4, 5, 3, 4, 7, 6, 8, 4, 2, 3, 7, 2, 7, 6, 2, 5, 8, 2, 4, 4, 5, 5, 3, 6, 3, 5, 6, 6, 3, 6, 7, 3],
    "
}

# Función para plotear el histograma y el ajuste
def plot_histogram(dataset, m, hist_color, fit_color, mean_color, std_dev_color):
    # Obtener los datos
    data = listas[dataset][:m]
    
    # Calcular la media y desviación estándar
    mean = np.mean(data)
    std_dev = np.std(data)
    
    # Plotear el histograma
    plt.hist(data, bins=np.arange(min(data), max(data)+1)-0.5, density=True, alpha=0.6, color=hist_color, edgecolor='black', linewidth=1.2, label='Datos experimentales')
    
    # Ajustar la distribución binomial
    x = np.arange(0, max(data)+1)
    n = len(x)
    p = mean / n
    y = binom.pmf(x, n, p)
    
    # Plotear el ajuste
    plt.plot(x, y, 'r--', linewidth=1.5, label=f'Ajuste Binomial\nMedia: {mean:.2f}\nDesviación Estándar: {std_dev:.2f}')
    
    # Graficar la media y la desviación estándar
    plt.axvline(x=min(data), color=mean_color, linestyle='-', linewidth=2, label=f'Valor mínimo: {min(data)}')
    plt.axvline(x=mean, color=std_dev_color, linestyle='-', linewidth=2, label=f'Desviación estándar: {std_dev:.2f}')

    plt.xlabel('Número de Caras')
    plt.ylabel('Densidad de probabilidad')
    plt.title(f'Histograma y Ajuste Binomial para los primeros {m} tiros del conjunto de datos "{dataset}"')
    plt.legend()
    plt.grid(True)
    st.pyplot()

# Crear la interfaz de usuario con Streamlit
def main():
    st.title('Ajuste Binomial e Histograma Interactivo')
    dataset = st.selectbox('Selecciona un conjunto de datos:', list(listas.keys()))
    m = st.slider('Selecciona el valor de m:', 1, 100, 10)
    hist_color = st.color_picker('Color del histograma:', '#00f')
    fit_color = st.color_picker('Color del ajuste:', '#f00')
    mean_color = st.color_picker('Color del valor mínimo:', '#0f0')
    std_dev_color = st.color_picker('Color de la desviación estándar:', '#ffa500')

    plot_histogram(dataset, m, hist_color, fit_color, mean_color, std_dev_color)

if __name__ == '__main__':
    main()
