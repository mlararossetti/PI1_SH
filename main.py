from fastapi import FastAPI, HTTPException
from typing import List, Dict
from pydantic import BaseModel
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = FastAPI()
#Rutas
csv_games = (r"/opt/render/project/src/dfgames.csv")
csv_reviews = (r"/opt/render/project/src/dfreviews.csv")
csv_items = (r"/opt/render/project/src/dfitems.parquet")
# Normalizar títulos por si el usuario ingresa en min, may
def normalize_title(title):
    return re.sub(r'[^a-zA-Z0-9 ]', ' ', title).lower().strip()

# Definir un modelo Pydantic para la respuesta
class GameRecommendation(BaseModel):
    title: str

@app.get("/")
def read_root():
    return {
        "message": (
            "Bienvenido a la API de Juegos. Contamos con 5 funciones para obtener información de tiempo real, "
            "tanto de desarrolladores y su popularidad como de usuarios, y géneros de videojuegos.\n"
            "A lo último contarás con un modelo de recomendación item-item, donde, con tu videojuego de preferencia, "
            "se te recomendarán otros similares."
        )
    }

@app.get("/developer/{desarrollador}")
def developer(desarrollador: str):
    """
    Obtiene un resumen de los juegos de un desarrollador específico,
    incluyendo la cantidad de juegos lanzados y el porcentaje de juegos gratuitos por año.
    """
    try:
        # Filtrar el DataFrame
        dfgames = pd.read_csv(csv_games, usecols=['id', 'price','developer','release_date'])
        juegos_desarrollador = dfgames[dfgames['developer'] == desarrollador].copy()

        # Verificar si hay juegos para el desarrollador
        if juegos_desarrollador.empty:
            raise HTTPException(status_code=404, detail="Desarrollador no encontrado")

        # Convertir la columna de fechas a datetime
        juegos_desarrollador['release_date'] = pd.to_datetime(juegos_desarrollador['release_date'], errors='coerce')

        if juegos_desarrollador['release_date'].dtype == 'datetime64[ns]':
            # Buscar el año de la fecha del lanzamiento del juego
            juegos_desarrollador['Año'] = juegos_desarrollador['release_date'].dt.year

        # Agrupar los juegos por año de lanzamiento
        resumen_anual = juegos_desarrollador.groupby('Año').agg(
            total_items=('id', 'count'),                      
            free_items=('price', lambda x: (x == 0).sum())   
        ).reset_index()

        # Calcular el porcentaje de juegos gratuitos por año
        resumen_anual['free_percentage'] = (resumen_anual['free_items'] / resumen_anual['total_items']) * 100

        # Redondear el porcentaje a 2 decimales
        resumen_anual['free_percentage'] = resumen_anual['free_percentage'].round(2)

        # Renombrar columnas para coincidir con la consigna
        resumen_anual.columns = ['Año', 'Cantidad de Items', 'Items Gratuitos', 'Porcentaje Gratuito']

        # Devolver solo las columnas seleccionadas como un diccionario
        return resumen_anual[['Año', 'Cantidad de Items', 'Porcentaje Gratuito']].to_dict(orient='records')
    
    except Exception as e:
        print(f"Error: {str(e)}")  # Imprimir el error en la consola para depuración
        raise HTTPException(status_code=500, detail="Error interno en el servidor")

@app.get("/user/{user_id}")
def user_data(user_id: str):
    """
    Obtiene información sobre un usuario específico, incluyendo el dinero gastado,
    la cantidad de ítems comprados y el porcentaje de recomendaciones positivas.
    """
    # Unir df necesarios
    dfgames = pd.read_csv(csv_games, usecols=['id', 'price'])
    dfreviews = pd.read_csv(csv_reviews, on_bad_lines='skip', usecols=['item_id', 'user_id', 'recommend'])
    merged_df = pd.merge(dfreviews[['user_id', 'item_id', 'recommend']], dfgames[['id', 'price']], left_on='item_id', right_on='id', how='left')

    # Filtrar por el user_id específico
    user_datas = merged_df[merged_df['user_id'] == user_id]

    if user_datas.empty:
        raise HTTPException(status_code=404, detail=f"No se encontraron datos para el usuario con ID: {user_id}")

    # Calcular el total gastado por el usuario
    total_spent = user_datas['price'].sum()

    # Calcular la cantidad de ítems comprados por el usuario
    item_count = user_datas['item_id'].count()

    # Calcular el porcentaje de recomendaciones
    total_recommendations = user_datas['recommend'].count()
    positive_recommendations = user_datas['recommend'].sum()  # Asumiendo que 'recommend' es True o False

    recommend_percentage = (positive_recommendations / total_recommendations * 100) if total_recommendations > 0 else 0

    # Crear un diccionario con los resultados
    results = {
        "Usuario X": user_id,
        "Dinero Gastado": float(total_spent),
        "% de recomendacion": float(round(recommend_percentage, 2)),
        "Cantidad de items": int(item_count)
    }

    return results

@app.get("/genre/{genero}")
def userForGenre(genero: str):
    """
    Obtiene el usuario que más horas ha jugado en un género específico,
    junto con las horas jugadas por año.
    """
    genero_normalizado = genero.lower()

    try:
        # Cargar los datos de los usuarios y el tiempo de juego
        dfitems = pd.read_parquet(csv_items, columns=['user_id', 'item_id', 'playtime_forever'])
        dfgames = pd.read_csv(csv_games, on_bad_lines='skip', usecols=['id', 'year', 'genres'])

        # Unir los DataFrames
        merged_df_ufg = pd.merge(
            dfitems[['item_id', 'user_id', 'playtime_forever']],
            dfgames[['id', 'genres', 'year']],
            left_on='item_id', 
            right_on='id', 
            how='left'
        )

        # Filtrar por género
        genre_data = merged_df_ufg[merged_df_ufg['genres'].str.contains(genero_normalizado, case=False, na=False)].copy()

        # Verificar si hay datos para el género especificado
        if genre_data.empty:
            raise HTTPException(status_code=404, detail=f"No se encontraron datos para el género especificado: {genero}")

        genre_data['playtime_forever'] = pd.to_numeric(genre_data['playtime_forever'], errors='coerce')

        # Agrupar por usuario y año, sumando las horas jugadas
        df_agrupado = genre_data.groupby(['user_id', 'year'])['playtime_forever'].sum().reset_index()

        # Encontrar el usuario con más horas jugadas en total
        horas_por_usuario = df_agrupado.groupby('user_id')['playtime_forever'].sum().reset_index()
        
        # Verificar si hay usuarios
        if horas_por_usuario.empty:
            raise HTTPException(status_code=404, detail="No se encontraron usuarios para el género especificado.")

        usuario_max = horas_por_usuario.loc[horas_por_usuario['playtime_forever'].idxmax()]

        # Obtener las horas jugadas por año para ese usuario
        horas_por_ano_usuario = df_agrupado[df_agrupado['user_id'] == usuario_max['user_id']]

        # Formatear el resultado en el formato deseado
        resultado = {
            f"Usuario con más horas jugadas para el género '{genero_normalizado}'": usuario_max['user_id'],
            "Horas jugadas": [{"Año": int(row['year']), "Horas": row['playtime_forever'] / 60} for _, row in horas_por_ano_usuario.iterrows()]  # Convertir minutos a horas
        }

        return resultado
    
    except Exception as e:
        # Manejo de excepciones para depurar
        print(f"Error en el endpoint /genre: {str(e)}")  # Imprimir el error para depuración
        raise HTTPException(status_code=500, detail="Error interno en el servidor.")


@app.get("/best_developer_year/{year}", response_model=List[Dict[str, str]])
def best_developer_year(year: int):
    """
    Obtiene el top 3 de desarrolladores con más recomendaciones en un año específico.
    """
    try:
        # Unir los dataframes
        dfgames = pd.read_csv(csv_games, on_bad_lines='skip', usecols=['id', 'year', 'developer'])
        dfreviews = pd.read_csv(csv_reviews, on_bad_lines='skip', usecols=['user_id', 'item_id', 'recommend', 'sentiment_analysis'])

        merged_df_dev = pd.merge(
            dfreviews[['user_id', 'item_id', 'recommend', 'sentiment_analysis']],
            dfgames[['id', 'developer', 'year']],
            left_on='item_id', right_on='id', how='left'
        )

        # Filtrar los datos del año solicitado
        year_data = merged_df_dev[merged_df_dev['year'] == year].copy()

        # Comprobar si hay datos disponibles para el año solicitado
        if year_data.empty:
            raise HTTPException(status_code=404, detail=f"No se encontraron datos para el año especificado: {year}")

        # Filtrar recomendaciones y análisis de sentimiento con valor 2
        filtered_df = year_data[(year_data['recommend'] == True) & (year_data['sentiment_analysis'] == 2)]

        # Contar recomendaciones por desarrollador
        developer_counts = filtered_df['developer'].value_counts().reset_index()
        developer_counts.columns = ['developer', 'recommendation_count']

        # Obtener el top 3 de desarrolladores con más recomendaciones
        top_developers = developer_counts.head(3)

        # Formatear el resultado
        resultado = [{"Puesto {}".format(index + 1): row['developer']} for index, row in top_developers.iterrows()]

        return resultado

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/developer_reviews/{desarrolladora}")
def developer_reviews_analysis(desarrolladora: str):
    """
    Analiza las reseñas de un desarrollador específico, devolviendo la cantidad de reseñas
    positivas y negativas.
    """
    # Unir los DataFrames necesarios
    dfgames = pd.read_csv(csv_games, on_bad_lines='skip', usecols=['id','developer'])
    dfreviews = pd.read_csv(csv_reviews, on_bad_lines='skip', usecols=['user_id','item_id','sentiment_analysis'])
    merged_df_dev2 = pd.merge(
        dfreviews[['item_id', 'sentiment_analysis']],
        dfgames[['id', 'developer']],
        left_on='item_id',
        right_on='id',
        how='left'
    )

    # Filtrar por el desarrollador específico
    desarrollador_data = merged_df_dev2[merged_df_dev2['developer'] == desarrolladora].copy()

    if desarrollador_data.empty:
        return {desarrolladora: {'Negative': 0, 'Positive': 0}}  # No hay reseñas

    # Contar la cantidad de reseñas positivas y negativas
    positive_count = (desarrollador_data['sentiment_analysis'] == 2).sum()
    negative_count = (desarrollador_data['sentiment_analysis'] == 0).sum()

    # Crear el diccionario con los resultados
    resultado = {
        desarrolladora: {
            'Negative': int(negative_count),
            'Positive': int(positive_count)
        }
    }

    return resultado

@app.get("/recommend/", response_model=List[GameRecommendation])
async def recommend_games(title: str):
    """
    Recomienda juegos similares a un título específico utilizando un modelo KNN.
    """
    # Cargar el CSV dentro de la función
    dfgames = pd.read_csv(('/opt/render/project/src/dfmodelo.csv'))
    
    # Crear una columna de títulos normalizados
    dfgames['normalized_title'] = dfgames['title'].apply(normalize_title)

    normalized_title = normalize_title(title)

    if normalized_title not in dfgames['normalized_title'].values:
        raise HTTPException(status_code=404, detail=f"El juego '{title}' no fue encontrado en la base de datos.")

    # Vectorizar e iniciar el modelo KNN
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(dfgames['description'])

    knn_model = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
    knn_model.fit(tfidf_matrix)

    game_index = dfgames[dfgames['normalized_title'] == normalized_title].index[0]

    distances, indices = knn_model.kneighbors(tfidf_matrix[game_index], n_neighbors=6)

    recommended_games = dfgames.iloc[indices.flatten()[1:]]

    # Crear una lista de recomendaciones
    recommendations = [
        GameRecommendation(title=row['title'])
        for _, row in recommended_games.iterrows()
    ]
    
    return recommendations
