from fastapi import FastAPI, HTTPException
import pandas as pd
from typing import List, Dict
from pydantic import BaseModel
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = FastAPI()

# Cargar los datos necesarios
dfgames = pd.read_csv('dfgames.csv')
dfreviews = pd.read_csv('dfreviews.csv')
dfitems = pd.read_parquet('dfitems.parquet')

# Normalizar títulos por si el usuario ingresa en min, may
def normalize_title(title: str) -> str:
    """Normaliza el título eliminando caracteres especiales y convirtiendo a minúsculas."""
    return re.sub(r'[^a-zA-Z0-9 ]', ' ', title).lower().strip()

# Vectorizar e iniciar el modelo KNN
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(dfgames['description'])

knn_model = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
knn_model.fit(tfidf_matrix)

# Definir un modelo Pydantic para la respuesta de recomendaciones
class GameRecommendation(BaseModel):
    title: str

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Juegos. Usa /developer/{desarrollador}, /user/{user_id}, /genre/{genero} o /best_developer_year/{año} para obtener información."}

@app.get("/developer/{desarrollador}")
def developer(desarrollador: str):
    try:
        juegos_desarrollador = dfgames[dfgames['developer'] == desarrollador].copy()
        if juegos_desarrollador.empty:
            raise HTTPException(status_code=404, detail="Desarrollador no encontrado")

        juegos_desarrollador['release_date'] = pd.to_datetime(juegos_desarrollador['release_date'], errors='coerce')
        juegos_desarrollador['Año'] = juegos_desarrollador['release_date'].dt.year

        resumen_anual = juegos_desarrollador.groupby('Año').agg(
            total_items=('id', 'count'),                      
            free_items=('price', lambda x: (x == 0).sum())   
        ).reset_index()

        resumen_anual['free_percentage'] = (resumen_anual['free_items'] / resumen_anual['total_items']) * 100
        resumen_anual['free_percentage'] = resumen_anual['free_percentage'].round(2)

        resumen_anual.columns = ['Año', 'Cantidad de Items', 'Items Gratuitos', 'Porcentaje Gratuito']
        return resumen_anual[['Año', 'Cantidad de Items', 'Porcentaje Gratuito']].to_dict(orient='records')
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno en el servidor")

@app.get("/user/{user_id}")
def user_data(user_id: str):
    merged_df = pd.merge(dfreviews[['user_id','item_id','recommend']], dfgames[['id', 'price']], left_on='item_id', right_on='id', how='left')
    user_datas = merged_df[merged_df['user_id'] == user_id]

    if user_datas.empty:
        raise HTTPException(status_code=404, detail=f"No se encontraron datos para el usuario con ID: {user_id}")

    total_spent = user_datas['price'].sum()
    item_count = user_datas['item_id'].count()
    total_recommendations = user_datas['recommend'].count()
    positive_recommendations = user_datas['recommend'].sum()

    recommend_percentage = (positive_recommendations / total_recommendations * 100) if total_recommendations > 0 else 0

    results = {
        "Usuario X": user_id,
        "Dinero Gastado": float(total_spent),
        "% de recomendacion": float(round(recommend_percentage, 2)),
        "Cantidad de items": int(item_count)
    }

    return results

@app.get("/genre/{genero}")
def userForGenre(genero: str):
    genero_normalizado = genero.lower()
    merged_df_ufg = pd.merge(dfitems[['item_id', 'user_id', 'playtime_forever']],
                              dfgames[['id', 'genres', 'release_date']],
                              left_on='item_id', right_on='id', how='left')

    genre_data = merged_df_ufg[merged_df_ufg['genres'].str.contains(genero_normalizado, case=False, na=False)].copy()
    if genre_data.empty:
        raise HTTPException(status_code=404, detail=f"No se encontraron datos para el género especificado: {genero}")

    genre_data['release_date'] = pd.to_datetime(genre_data['release_date'], errors='coerce')
    genre_data['year'] = genre_data['release_date'].dt.year
    genre_data['playtime_forever'] = pd.to_numeric(genre_data['playtime_forever'], errors='coerce')

    df_agrupado = genre_data.groupby(['user_id', 'year'])['playtime_forever'].sum().reset_index()
    horas_por_usuario = df_agrupado.groupby('user_id')['playtime_forever'].sum().reset_index()
    usuario_max = horas_por_usuario.loc[horas_por_usuario['playtime_forever'].idxmax()]

    horas_por_ano_usuario = df_agrupado[df_agrupado['user_id'] == usuario_max['user_id']]
    resultado = {
        f"Usuario con más horas jugadas para el género '{genero_normalizado}'": usuario_max['user_id'],
        "Horas jugadas": [{"Año": int(row['year']), "Horas": row['playtime_forever'] / 60} for _, row in horas_por_ano_usuario.iterrows()]
    }

    return resultado

@app.get("/best_developer_year/{year}", response_model=List[Dict[str, str]])
def best_developer_year(year: int):
    try:
        merged_df_dev = pd.merge(dfreviews[['user_id', 'item_id', 'recommend', 'sentiment_analysis']],
                                 dfgames[['id', 'developer', 'release_date']],
                                 left_on='item_id', right_on='id', how='left')

        merged_df_dev['release_date'] = pd.to_datetime(merged_df_dev['release_date'], errors='coerce')
        merged_df_dev['year'] = merged_df_dev['release_date'].dt.year

        year_data = merged_df_dev[merged_df_dev['year'] == year].copy()
        if year_data.empty:
            raise HTTPException(status_code=404, detail=f"No se encontraron datos para el año especificado: {year}")

        filtered_df = year_data[(year_data['recommend'] == True) & (year_data['sentiment_analysis'] == 2)]
        developer_counts = filtered_df['developer'].value_counts().reset_index()
        developer_counts.columns = ['developer', 'recommendation_count']

        top_developers = developer_counts.head(3)
        resultado = [{"Puesto {}".format(index + 1): row['developer']} for index, row in top_developers.iterrows()]

        return resultado

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/developer_reviews/{desarrolladora}")
def developer_reviews_analysis(desarrolladora: str):
    merged_df_dev2 = pd.merge(
        dfreviews[['item_id', 'sentiment_analysis']],
        dfgames[['id', 'developer']],
        left_on='item_id',
        right_on='id',
        how='left'
    )

    desarrollador_data = merged_df_dev2[merged_df_dev2['developer'] == desarrolladora].copy()
    if desarrollador_data.empty:
        return {desarrolladora: {'Negative': 0, 'Positive': 0}}  # No hay reseñas

    positive_count = (desarrollador_data['sentiment_analysis'] == 2).sum()
    negative_count = (desarrollador_data['sentiment_analysis'] == 0).sum()

    resultado = {
        desarrolladora: {
            'Negative': int(negative_count),
            'Positive': int(positive_count)
        }
    }

    return resultado

# Recomendar juegos según el modelo creado
@app.get("/recommend/", response_model=List[GameRecommendation])
async def recommend_games(title: str):
    normalized_title = normalize_title(title)
    
    if 'normalized_title' not in dfgames.columns:
        dfgames['normalized_title'] = dfgames['title'].apply(normalize_title)
    
    if normalized_title not in dfgames['normalized_title'].values:
        raise HTTPException(status_code=404, detail=f"El juego '{title}' no fue encontrado en la base de datos.")
    
    game_index = dfgames[dfgames['normalized_title'] == normalized_title].index[0]
    
    distances, indices = knn_model.kneighbors(tfidf_matrix[game_index], n_neighbors=6)  
    
    recommended_games = dfgames.iloc[indices.flatten()[1:]]  # Excluir el juego mismo
    
    recommendations = [GameRecommendation(title=row['title']) for _, row in recommended_games.iterrows()]
    
    return recommendations
