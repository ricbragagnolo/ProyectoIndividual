import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Cargar los datos procesados
movies_df = pd.read_csv('CSV/data_fastapi.csv')

# Resumen estadístico
print("Resumen estadístico de las variables numéricas:")
print(movies_df.describe())

# Valores nulos
print("\nCantidad de valores nulos en cada columna:")
print(movies_df.isnull().sum())

# Verificar la existencia de la columna 'revenue' antes de graficar
if 'revenue' in movies_df.columns:
    # Distribución de revenue
    plt.figure(figsize=(10, 6))
    sns.histplot(movies_df['revenue'], bins=50, kde=True)
    plt.title('Distribución de Revenue')
    plt.xlabel('Revenue')
    plt.ylabel('Frecuencia')
    plt.show()
else:
    print("\nLa columna 'revenue' no está presente en el DataFrame.")

# Verificar la existencia de la columna 'budget' antes de graficar
if 'budget' in movies_df.columns:
    # Distribución de budget
    plt.figure(figsize=(10, 6))
    sns.histplot(movies_df['budget'], bins=50, kde=True)
    plt.title('Distribución de Budget')
    plt.xlabel('Budget')
    plt.ylabel('Frecuencia')
    plt.show()
else:
    print("\nLa columna 'budget' no está presente en el DataFrame.")

# Distribución de años de lanzamiento
plt.figure(figsize=(10, 6))
sns.countplot(data=movies_df, x='release_year')
plt.title('Distribución de Años de Lanzamiento')
plt.xlabel('Año de Lanzamiento')
plt.ylabel('Cantidad de Películas')
plt.xticks(rotation=90)
plt.show()

# Correlación entre variables numéricas
plt.figure(figsize=(12, 8))
numeric_df = movies_df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Mapa de Calor de Correlaciones')
plt.show()

# Nube de palabras de los títulos de las películas
title_corpus = ' '.join(movies_df['title'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(title_corpus)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de Palabras de los Títulos de las Películas')
plt.show()
