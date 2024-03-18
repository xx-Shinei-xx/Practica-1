import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

# datos de las caras de las monedas
listas = {
     "Guillermo y Shawn": [2, 4, 5, 3, 7, 3, 4, 6, 4, 4, 3, 5, 3, 2, 3, 4, 8, 6, 4, 2, 5, 5, 3, 8, 4, 7, 4, 6, 3, 5, 8, 7, 3, 3, 6, 5, 4, 4, 5, 2, 5, 3, 7, 6, 3, 6, 5, 2, 4, 6, 5, 4, 6, 3, 6, 5, 3, 7, 8, 7, 4, 4, 4, 8, 5, 4, 3, 5, 7, 5, 2, 2, 3, 5, 1, 6, 4, 6, 4, 4, 3, 3, 6, 6, 3, 4, 5, 5, 5, 7, 6, 7, 4, 3, 5, 4, 5, 7, 6, 5],
    "Lobsang y Rebeca": [6, 5, 5, 6, 5, 4, 6, 6, 5, 4, 6, 6, 5, 6, 9, 8, 1, 7, 5, 3, 5, 3, 3, 4, 3, 5, 4, 4, 6, 2, 5, 6, 7, 5, 5, 2, 3, 5, 7, 6, 5, 1, 6, 4, 4, 8, 5, 3, 6, 5, 6, 4, 5, 5, 3, 2, 6, 5, 2, 9, 7, 4, 7, 4, 3, 3, 6, 6, 4, 4, 6, 5, 5, 4, 6, 4, 9, 6, 4, 4, 8, 6, 4, 4, 8, 6, 8, 3, 6, 2, 5, 6, 2, 4, 5, 3, 4, 6, 5, 7],
    "Diego y Saul": [5, 6, 5, 5, 4, 4, 4, 3, 6, 5, 4, 7, 5, 7, 3, 5, 4, 7, 3, 4, 6, 3, 4, 5, 6, 2, 7, 3, 6, 2, 4, 7, 5, 5, 5, 3, 6, 6, 5, 4, 4, 7, 4, 7, 6, 5, 4, 4, 3, 5, 5, 4, 4, 7, 4, 5, 5, 4, 7, 6, 9, 5, 5, 5, 4, 5, 5, 7, 5, 4, 8, 3, 4, 4, 4, 8, 4, 9, 7, 7, 5, 5, 7, 5, 4, 4, 6, 7, 4, 2, 5, 5, 3, 6, 7, 5, 4, 4, 4, 7],
    "Giovanna y Mario": [5, 5, 5, 7, 7, 4, 7, 4, 6, 6, 6, 5, 5, 5, 6, 6, 6, 6, 7, 4, 5, 4, 5, 6, 5, 8, 7, 4, 3, 6, 4, 3, 6, 2, 7, 5, 8, 7, 6, 7, 4, 5, 5, 6, 4, 7, 4, 6, 4, 3, 4, 5, 5, 4, 3, 5, 6, 7, 5, 4, 5, 4, 4, 4, 6, 8, 6, 7, 5, 1, 3, 6, 4, 5, 4, 3, 5, 4, 3, 4, 6, 8, 5, 6, 5, 7, 5, 4, 6, 5, 4, 4, 10, 8, 3, 7, 5, 5, 4, 4],
    "Dessiré y Fabricio": [6, 4, 3, 6, 6, 6, 6, 7, 4, 4, 5, 4, 5, 3, 4, 8, 5, 3, 6, 6, 6, 5, 5, 5, 6, 4, 6, 6, 7, 6, 6, 6, 5, 4, 2, 5, 3, 6, 4, 4, 6, 5, 3, 4, 5, 5, 6, 5, 7, 5, 3, 3, 5, 5, 5, 4, 10, 5, 6, 4, 3, 5, 6, 4, 3, 4, 6, 5, 4, 6, 8, 5, 5, 5, 4, 5, 8, 4, 5, 5, 3, 3, 4, 5, 2, 7, 4, 5, 4, 6, 5, 6, 3, 6, 5, 7, 7, 9, 5, 3],
    "Jacobo y Cesar": [6, 6, 5, 3, 2, 5, 7, 8, 4, 5, 3, 4, 7, 6, 8, 4, 2, 3, 7, 2, 7, 6, 2, 5, 8, 2, 4, 4, 5, 5, 3, 6, 3, 5, 6, 6, 3, 6, 7, 3, 5, 4, 5, 4, 3, 5, 6, 4, 7, 4, 7, 6, 4, 6, 7, 6, 7, 4, 2, 4, 3, 4, 5, 5, 7, 4, 5, 4, 2, 4, 7, 5, 3, 5, 5, 4, 4, 6, 5, 4, 4, 4, 5, 4, 6, 6, 6, 8, 3, 5, 7, 3, 4, 8, 4, 6, 5, 4, 6, 4],
}

@st.cache
def get_histogram_data(dataset, m):
    data = listas[dataset][:m]
    mean = np.mean(data)
    std_dev = np.std(data)
    return data, mean, std_dev

def plot_histogram(data, mean, std_dev, hist_color, fit_color, mean_color, std_dev_color):
    # Plotear el histograma
    plt.hist(data, bins=np.arange(min(data), max(data)+2)-0.5, density=True, alpha=0.6, color=hist_color, edgecolor='black', linewidth=1.2, label='Datos experimentales')

    # fit de la distribución binomial
    fitted_results = ss.fit(ss.binom, data, bounds=[(0, 100), (0, 1)])
    n, p = fitted_results[0], fitted_results[2]

    x = np.arange(min(data), max(data)+1)
    y = ss.binom.pmf(x, n, p)

    # Plotear el ajuste
    plt.plot(x, y, 'r--', linewidth=1.5, label=f'Ajuste Binomial\nMedia: {mean:.2f}\nDesviación Estándar: {std_dev:.2f}')

    # Graficar la media y la desviación estándar
    plt.axvline(x=min(data), color=mean_color, linestyle='-', linewidth=2, label=f'Valor mínimo: {min(data)}')
    plt.axvline(x=mean, color=std_dev_color, linestyle='-', linewidth=2, label=f'Desviación estándar: {std_dev:.2f}')

    plt.xlabel('Número de Caras')
    plt.ylabel('Probabilidad')
    plt.title(f'Histograma y Ajuste Binomial para los primeros {m} tiros de la pareja de "{dataset}"')
    plt.legend()
    plt.grid(True)
    st.pyplot()

# Encabezado
st.title('Análisis de lanzamientos de monedas')

# widgets
dataset = st.selectbox('Datos de la pareja:', options=list(listas.keys()))
m = st.slider('Valor de m:', min_value=1, max_value=100, step=1, value=10)
hist_color = st.color_picker('Color del histograma:', value='blue')
fit_color = st.color_picker('Color del fit:', value='red')
mean_color = st.color_picker('Color del valor mínimo:', value='green')
std_dev_color = st.color_picker('Color de la desviación estándar:', value='orange')

# Obtener los datos
data, mean, std_dev = get_histogram_data(dataset, m)

# plotear el histograma

plot_histogram(data, mean, std_dev, hist_color, fit_color, mean_color, std_dev_color)


if _name_ == '_main_':
    main()
