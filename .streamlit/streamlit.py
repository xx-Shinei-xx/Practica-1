import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import streamlit as st
from IPython.display import YouTubeVideo
st.set_option('deprecation.showPyplotGlobalUse', False)



# Definir los datos
listas = {
    "Lobsang - Rebeca": [6, 5, 5, 6, 5, 4, 6, 6, 5, 4, 6, 6, 5, 6, 9, 8, 1, 7, 5, 3, 5, 3, 3, 4, 3, 5, 4, 4, 6, 2, 5, 6, 7, 5, 5, 2, 3, 5, 7, 6, 5, 1, 6, 4, 4, 8, 5, 3, 6, 5, 6, 4, 5, 5, 3, 2, 6, 5, 2, 9, 7, 4, 7, 4, 3, 3, 6, 6, 4, 4, 6, 5, 5, 4, 6, 4, 9, 6, 4, 4, 8, 6, 4, 4, 8, 6, 8, 3, 6, 2, 5, 6, 2, 4, 5, 3, 4, 6, 5, 7],
    "Diego - Saul": [5, 6, 5, 5, 4, 4, 4, 3, 6, 5, 4, 7, 5, 7, 3, 5, 4, 7, 3, 4, 6, 3, 4, 5, 6, 2, 7, 3, 6, 2, 4, 7, 5, 5, 5, 3, 6, 6, 5, 4, 4, 7, 4, 7, 6, 5, 4, 4, 3, 5, 5, 4, 4, 7, 4, 5, 5, 4, 7, 6, 9, 5, 5, 5, 4, 5, 5, 7, 5, 4, 8, 3, 4, 4, 4, 8, 4, 9, 7, 7, 5, 5, 7, 5, 4, 4, 6, 7, 4, 2, 5, 5, 3, 6, 7, 5, 4, 4, 4, 7],
    "Giovanna - Mario": [5, 5, 5, 7, 7, 4, 7, 4, 6, 6, 6, 5, 5, 5, 6, 6, 6, 6, 7, 4, 5, 4, 5, 6, 5, 8, 7, 4, 3, 6, 4, 3, 6, 2, 7, 5, 8, 7, 6, 7, 4, 5, 5, 6, 4, 7, 4, 6, 4, 3, 4, 5, 5, 4, 3, 5, 6, 7, 5, 4, 5, 4, 4, 4, 6, 8, 6, 7, 5, 1, 3, 6, 4, 5, 4, 3, 5, 4, 3, 4, 6, 8, 5, 6, 5, 7, 5, 4, 6, 5, 4, 4, 10, 8, 3, 7, 5, 5, 4, 4],
    "Dessiré - Fabricio": [6, 4, 3, 6, 6, 6, 6, 7, 4, 4, 5, 4, 5, 3, 4, 8, 5, 3, 6, 6, 6, 5, 5, 5, 6, 4, 6, 6, 7, 6, 6, 6, 5, 4, 2, 5, 3, 6, 4, 4, 6, 5, 3, 4, 5, 5, 6, 5, 7, 5, 3, 3, 5, 5, 5, 4, 10, 5, 6, 4, 3, 5, 6, 4, 3, 4, 6, 5, 4, 6, 8, 5, 5, 5, 4, 5, 8, 4, 5, 5, 3, 3, 4, 5, 2, 7, 4, 5, 4, 6, 5, 6, 3, 6, 5, 7, 7, 9, 5, 3],
    "Jacobo - Cesar": [6, 6, 5, 3, 2, 5, 7, 8, 4, 5, 3, 4, 7, 6, 8, 4, 2, 3, 7, 2, 7, 6, 2, 5, 8, 2, 4, 4, 5, 5, 3, 6, 3, 5, 6, 6, 3, 6, 7, 3, 5, 4, 5, 4, 3, 5, 6, 4, 7, 4, 7, 6, 4, 6, 7, 6, 7, 4, 2, 4, 3, 4, 5, 5]
}

# Función para mostrar el histograma y el video de YouTube
def plot_histogram_and_video(dataset, m, hist_color, fit_color, mean_color, std_dev_color, youtube_link):
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
    
    # Añadir cuadro de texto
    textstr = f'Valor mínimo: {min(data)}\nDesviación estándar: {std_dev:.2f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=props)

    plt.xlabel('Número de Caras')
    plt.ylabel('Densidad de probabilidad')
    plt.title(f'Histograma y Ajuste Binomial para los primeros {m} tiros del conjunto de datos "{dataset}"')
    plt.legend()
    plt.grid(True)
    
    # Mostrar el video de YouTube
    if youtube_link:
        st.write("Video de YouTube:")
        st.write(YouTubeVideo(youtube_link))

    st.pyplot()

# Interfaz de usuario interactiva
dataset = st.sidebar.selectbox('Conjunto de datos:', list(listas.keys()))
m = st.sidebar.slider('m:', min_value=1, max_value=100, step=1, value=10)
hist_color = st.sidebar.color_picker('Color del histograma:', value='blue')
fit_color = st.sidebar.color_picker('Color del ajuste:', value='red')
mean_color = st.sidebar.color_picker('Color del valor mínimo:', value='green')
std_dev_color = st.sidebar.color_picker('Color de la desviación estándar:', value='orange')
youtube_link = st.sidebar.text_input('Enlace de YouTube:', '')

plot_histogram_and_video(dataset, m, hist_color, fit_color, mean_color, std_dev_color, youtube_link)
