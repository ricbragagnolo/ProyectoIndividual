import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import os

app = FastAPI()

# Definir la ruta relativa al archivo data_fastapi.csv
base_dir = os.path.dirname(__file__)
filepath = os.path.join(base_dir, "CSV/data_fastapi.csv")

# Cargar el archivo procesado
movies_df = pd.read_csv(filepath, low_memory=False)
movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')


@app.get("/")
def read_root():
    return {"Mensaje": "Bienvenido a la API de Ricardo Bragagnolo"}

@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    meses = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 
        'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8, 
        'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    
    mes_num = meses.get(mes.lower())
    if mes_num is None:
        raise HTTPException(status_code=400, detail="Mes no válido")

    count = movies_df[movies_df['release_date'].dt.month == mes_num].shape[0]
    return {"Mensaje": f"{count} cantidad de películas fueron estrenadas en el mes de {mes}"}

@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    dias = {
        'lunes': 0, 'martes': 1, 'miércoles': 2, 'jueves': 3, 
        'viernes': 4, 'sábado': 5, 'domingo': 6
    }

    dia_num = dias.get(dia.lower())
    if dia_num is None:
        raise HTTPException(status_code=400, detail="Día no válido")

    count = movies_df[movies_df['release_date'].dt.dayofweek == dia_num].shape[0]
    return {"Mensaje": f"{count} cantidad de películas fueron estrenadas en los días {dia}"}

@app.get("/score_titulo/{titulo}")
def score_titulo(titulo: str):
    pelicula = movies_df[movies_df['title'].str.lower() == titulo.lower()]
    
    if pelicula.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada")

    titulo = pelicula.iloc[0]['title']
    anio = pelicula.iloc[0]['release_year']
    score = pelicula.iloc[0]['vote_average']
    return {"Mensaje": f"La película {titulo} fue estrenada en el año {anio} con un score/popularidad de {score}"}

@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo: str):
    pelicula = movies_df[movies_df['title'].str.lower() == titulo.lower()]

    if pelicula.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada")

    titulo = pelicula.iloc[0]['title']
    anio = pelicula.iloc[0]['release_year']
    votos = pelicula.iloc[0]['vote_count']
    promedio = pelicula.iloc[0]['vote_average']

    if votos < 2000:
        return {"Mensaje": f"La película {titulo} no cumple con la condición de tener al menos 2000 valoraciones"}

    return {"Mensaje": f"La película {titulo} fue estrenada en el año {anio}. La misma cuenta con un total de {votos} valoraciones, con un promedio de {promedio}"}

@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    actor_films = movies_df[movies_df['cast_names'].apply(lambda x: nombre_actor.lower() in str(x).lower())]

    if actor_films.empty:
        raise HTTPException(status_code=404, detail="Actor no encontrado")

    cantidad_filmaciones = actor_films.shape[0]
    exito_total = actor_films['return'].sum()
    promedio_retorno = actor_films['return'].mean()
    
    return {"Mensaje": f"El actor {nombre_actor} ha participado de {cantidad_filmaciones} cantidad de filmaciones, el mismo ha conseguido un retorno de {exito_total} con un promedio de {promedio_retorno} por filmación"}

@app.get("/get_director/{nombre_director}")
def get_director(nombre_director: str):
    director_films = movies_df[movies_df['crew_names'].apply(lambda x: nombre_director.lower() in str(x).lower())]

    if director_films.empty:
        raise HTTPException(status_code=404, detail="Director no encontrado")

    films = []
    for index, row in director_films.iterrows():
        films.append({
            "Titulo": row['title'],
            "Fecha_lanzamiento": row['release_date'],
            "Retorno": row['return'],
            "Costo": row['budget'],
            "Ganancia": row['revenue']
        })

    exito_total = director_films['return'].sum()

    return {"Mensaje": f"El director {nombre_director} ha conseguido un retorno de {exito_total}.",
            "peliculas": films}

@app.get("/recomendacion/{titulo}")
def recomendacion(titulo: str):
    pelicula = movies_df[movies_df['title'].str.lower() == titulo.lower()]

    if pelicula.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada")

    cv = CountVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(movies_df['title'])

    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    idx = movies_df[movies_df['title'].str.lower() == titulo.lower()].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    movie_indices = [i[0] for i in sim_scores]
    recommendations = movies_df['title'].iloc[movie_indices].tolist()

    return {"Recomendación": recommendations}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
