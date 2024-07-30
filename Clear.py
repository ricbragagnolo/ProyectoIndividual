import pandas as pd
import json
import os

def corregir_json(val):
    """
    Corrige el formato JSON de un valor de cadena.
    """
    try:
        return json.loads(val.replace("'", "\"").replace("None", "null"))
    except json.JSONDecodeError:
        return None

def desanidar_diccionario(df, campo):
    """
    Desanida campos que son diccionarios en un DataFrame.
    """
    if campo in df.columns and df[campo].dtype == object:
        df[campo] = df[campo].fillna('{}').apply(corregir_json)
        
        if isinstance(df[campo].iloc[0], dict):
            temp_df = df[campo].apply(pd.Series).add_suffix(f'_{campo}')
            df = df.drop(campo, axis=1).join(temp_df)
    return df

def desanidar_lista_diccionarios(df, campo):
    """
    Desanida campos que son listas de diccionarios en un DataFrame.
    """
    if campo in df.columns and df[campo].dtype == object:
        df[campo] = df[campo].fillna('[]').apply(corregir_json)
        
        def extract_info(lista, key):
            if isinstance(lista, list):
                return ', '.join([str(item[key]) for item in lista if key in item])
            return ''
        
        df[f'{campo}_names'] = df[campo].apply(lambda x: extract_info(x, 'name'))
        df[f'{campo}_ids'] = df[campo].apply(lambda x: extract_info(x, 'id'))
        df = df.drop(campo, axis=1)
    return df

def preparar_datos(movies_filepath, credits_filepath):
    """
    Prepara los datos cargando, limpiando y desanidando los campos necesarios.
    """
    # Cargar los archivos movies_dataset.csv y credits.csv con la opción low_memory=False para evitar el DtypeWarning
    movies_df = pd.read_csv(movies_filepath, low_memory=False)
    credits_df = pd.read_csv(credits_filepath, low_memory=False)

    # Convertir la columna 'id' a tipo int64 en ambos DataFrames
    movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce').astype('Int64')
    credits_df['id'] = pd.to_numeric(credits_df['id'], errors='coerce').astype('Int64')

    # Eliminar columnas no utilizadas en movies_df
    columns_to_drop = ['video', 'imdb_id', 'adult', 'original_title', 'poster_path', 'homepage']
    movies_df = movies_df.drop(columns=columns_to_drop, axis=1)

    # Rellenar valores nulos en revenue y budget con 0
    movies_df['revenue'] = movies_df['revenue'].fillna(0)
    movies_df['budget'] = movies_df['budget'].fillna(0)

    # Convertir revenue y budget a tipo numérico (float)
    movies_df['revenue'] = pd.to_numeric(movies_df['revenue'], errors='coerce')
    movies_df['budget'] = pd.to_numeric(movies_df['budget'], errors='coerce')

    # Eliminar filas con valores nulos en release_date
    movies_df = movies_df.dropna(subset=['release_date'])

    # Asegurar que las fechas estén en el formato AAAA-mm-dd
    movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')
    movies_df['release_date'] = movies_df['release_date'].dt.strftime('%Y-%m-%d')

    # Crear columna release_year
    movies_df['release_year'] = pd.to_datetime(movies_df['release_date']).dt.year

    # Crear columna return (revenue / budget)
    movies_df['return'] = movies_df.apply(lambda row: row['revenue'] / row['budget'] if row['budget'] > 0 else 0, axis=1)

    # Desanidar los campos belongs_to_collection y production_companies
    movies_df = desanidar_diccionario(movies_df, 'belongs_to_collection')
    movies_df = desanidar_lista_diccionarios(movies_df, 'production_companies')

    # Desanidar los campos cast y crew en credits_df
    credits_df = desanidar_lista_diccionarios(credits_df, 'cast')
    credits_df = desanidar_lista_diccionarios(credits_df, 'crew')

    # Unir ambos DataFrames por la columna 'id'
    combined_df = pd.merge(movies_df, credits_df, on='id')

    return combined_df

def main():
    """
    Función principal para preparar los datos y guardarlos en un nuevo archivo CSV.
    """
    # Definir las rutas absolutas a los archivos movies_dataset.csv y credits.csv
    movies_filepath = os.path.join(os.path.dirname(__file__), '..', 'DATA', 'movies_dataset.csv')
    credits_filepath = os.path.join(os.path.dirname(__file__), '..', 'DATA', 'credits.csv')

    # Preparar los datos
    combined_df = preparar_datos(movies_filepath, credits_filepath)

    # Seleccionar solo las columnas necesarias
    selected_columns = ['release_date', 'title', 'release_year', 'vote_average', 'vote_count', 'cast_names', 'crew_names', 'return']
    filtered_df = combined_df[selected_columns]

    # Mostrar los primeros registros para verificar
    print("Primeros registros después de preparar los datos:")
    print(filtered_df.head())

    # Guardar el DataFrame filtrado en un nuevo archivo CSV
    output_filepath = os.path.join(os.path.dirname(__file__), 'CSV', 'data_fastapi.csv')
    filtered_df.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    main()
