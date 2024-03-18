import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

# Cargar datos del archivo CSV
@st.cache
def load_data():
    return pd.read_csv('Binomial-fichas.csv')

data = load_data()

st.title('Histograma y ajuste de distribución binomial')

# Selección de la cantidad de lanzamientos de monedas (m)
m = st.slider('Selecciona la cantidad de lanzamientos de monedas (m)', min_value=0, max_value=100, value=50)

# Seleccionar los primeros m lanzamientos de cada moneda
data_selected = data.loc[:, :'Coin_10'].head(m)

# Calcular el conteo de caras para cada conjunto de lanzamientos de monedas
counts = data_selected.sum(axis=1)

# Crear el histograma de los conteos de caras
fig, ax = plt.subplots()
ax.hist(counts, bins=np.arange(0, 11, 1), density=True, alpha=0.5, color='blue', label='Histograma')

# Ajustar una distribución binomial a los datos
fitted_results = ss.binom.fit(counts, loc=0, scale=1)

# Generar la distribución binomial ajustada
x = np.arange(0, 11, 1)
binomial_dist = ss.binom.pmf(x, *fitted_results)

# Graficar la distribución binomial ajustada
ax.plot(x, binomial_dist, 'r-', label='Distribución Binomial Ajustada')

# Mostrar los valores obtenidos del ajuste
st.write("Parámetros del ajuste de la distribución binomial (n, p):", fitted_results)

# Calcular el conteo medio de caras y su desviación estándar experimentalmente
mean_count = np.mean(counts)
std_count = np.std(counts)

st.write("Conteo medio de caras experimental:", mean_count)
st.write("Desviación estándar experimental del conteo de caras:", std_count)

# Calcular el conteo medio de caras y su desviación estándar obtenidos del ajuste
mean_count_fit = ss.binom.mean(*fitted_results)
std_count_fit = ss.binom.std(*fitted_results)

st.write("Conteo medio de caras obtenido del ajuste:", mean_count_fit)
st.write("Desviación estándar del conteo de caras obtenida del ajuste:", std_count_fit)

# Configuración del gráfico
ax.set_xlabel('Conteo de caras')
ax.set_ylabel('Densidad')
ax.set_title(f'Distribución del conteo de caras de los primeros {m} tiros de 10 monedas')
ax.legend()
ax.grid(True)

# Mostrar el gráfico en Streamlit
st.pyplot(fig)

if __name__ == "__main__":
    valores_de_n_y_p()
    
