
from fastapi import FastAPI, HTTPException
import pandas as pd
from typing import List, Dict


app = FastAPI()
csv_file_path = r'C:\Users\user\OneDrive\Escritorio\SOYHENRY\Curso Data Science\fastapi-app\dfgames.csv'
csv_file_path2 = r'C:\Users\user\OneDrive\Escritorio\SOYHENRY\Curso Data Science\fastapi-app\dfreviews.csv'
csv_file_path3 = r'C:\Users\user\OneDrive\Escritorio\SOYHENRY\Curso Data Science\fastapi-app\dfitems.parquet'
dfgames = pd.read_csv(csv_file_path)
dfreviews = pd.read_csv(csv_file_path2)
dfitems = pd.read_parquet(csv_file_path3)

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Juegos. Usa /developer/{desarrollador} , /user/{user_id}, /genre/{genero} o /best_developer/ {año} para obtener información."}

@app.get("/developer/{desarrollador}")
def developer(desarrollador: str):
    try:
        # Filtrar el DataFrame
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
    # Unir df necesarios
    merged_df = pd.merge(dfreviews[['user_id','item_id','recommend']], dfgames[['id', 'price']], left_on='item_id', right_on='id', how='left')

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
    # Normalizar el género a minúsculas para la búsqueda
    genero_normalizado = genero.lower()

    # Hacer merge de los DataFrames en función de 'item_id' y 'user_id'
    merged_df_ufg = pd.merge(dfitems[['item_id', 'user_id', 'playtime_forever']],
                              dfgames[['id', 'genres', 'release_date']],
                              left_on='item_id', right_on='id', how='left')

    # Filtrar por el género específico usando str.contains
    genre_data = merged_df_ufg[merged_df_ufg['genres'].str.contains(genero_normalizado, case=False, na=False)].copy()

    if genre_data.empty:
        raise HTTPException(status_code=404, detail=f"No se encontraron datos para el género especificado: {genero}")

    # Asegurarse de que la columna de fecha sea del tipo datetime
    genre_data['release_date'] = pd.to_datetime(genre_data['release_date'], errors='coerce')

    # Extraer el año de la columna de fecha
    genre_data['year'] = genre_data['release_date'].dt.year

    # Convertir 'playtime_forever' a numérico
    genre_data['playtime_forever'] = pd.to_numeric(genre_data['playtime_forever'], errors='coerce')

    # Agrupar por usuario y año, sumando las horas jugadas
    df_agrupado = genre_data.groupby(['user_id', 'year'])['playtime_forever'].sum().reset_index()

    # Encontrar el usuario con más horas jugadas en total
    horas_por_usuario = df_agrupado.groupby('user_id')['playtime_forever'].sum().reset_index()
    usuario_max = horas_por_usuario.loc[horas_por_usuario['playtime_forever'].idxmax()]

    # Obtener las horas jugadas por año para ese usuario
    horas_por_ano_usuario = df_agrupado[df_agrupado['user_id'] == usuario_max['user_id']]

    # Formatear el resultado en el formato deseado
    resultado = {
        f"Usuario con más horas jugadas para el género '{genero_normalizado}'": usuario_max['user_id'],
        "Horas jugadas": [{"Año": int(row['year']), "Horas": row['playtime_forever'] / 60} for _, row in horas_por_ano_usuario.iterrows()]  # Convertir minutos a horas
    }

    return resultado



@app.get("/best_developer_year/{year}", response_model=List[Dict[str, str]])
def best_developer_year(year: int):
    try:
        # Unir los dataframes `dfreviews` y `dfgames`
        merged_df_dev = pd.merge(dfreviews[['user_id', 'item_id', 'recommend', 'sentiment_analysis']],
                                 dfgames[['id', 'developer', 'release_date']],
                                 left_on='item_id', right_on='id', how='left')

        # Convertir `release_date` a tipo datetime
        merged_df_dev['release_date'] = pd.to_datetime(merged_df_dev['release_date'], errors='coerce')

        # Extraer el año de la fecha de lanzamiento
        merged_df_dev['year'] = merged_df_dev['release_date'].dt.year

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
    # Unir los DataFrames necesarios
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

