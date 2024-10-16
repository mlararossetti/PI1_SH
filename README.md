# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>

# <h1 align=center>**Machine Learning Operations (MLOps)**</h1>

# <h2 align=center>**Sistema de recomendación de juegos de la plataforma Steam**</h2> 

<p align="center">


## Datos  

Dataset en formatos JSON comprimidos (steam_games,user_reviews y users_items). Se cargaron los datos, transformandolos en un archivo de texto para su posterior lectura y se desanidaron las columnas anidadas que eran de relevancia. Luego del proceso, se exportaron tres archivos csv. 

## ETL

Se hizo una limpieza general de los tres documentos, eliminando columnas que no eran necesarias y que aumentaban el peso de los archivos, tales como en dfreviews columnas[**funny**,**posted**,**last_edited**,**user_url**].

Se eliminan filas enteras de valores nulos y duplicados.

Se pasaron columnas como **price** y **playtime_forever** a numérico, y **release_date** a formato de fecha, extrayendo el año de cada fila para poder utilizarlo posteriormente.

Se utilizó la libreria Vader Sentiment para poder crear una función de análisis de sentimiento utilizando el texto de `reviews`, categorizando 2 como positivo, 1 como neutro y 0 como negativo.

## EDA

Primero se visualizan los data frames para ver las columnas restantes, luego se cuentan los datos únicos que serian juegos en cada dataframe, para ver dónde hay mas datos, y cuantas reviews, y otra información extra tenemos.

Se analizan las variables numéricas como el precio de los videojuegos y la cantidad de minutos jugados.

Se hace un merge entre los distintos archivos para comparar por ejemplo popularidad segun año, género, etc.


<p align="center">
    <img src="images/genre.png" height="400">
</p>


Tambien se hicieron varias consultas para entender un poco mejor los datos como los juegos mas populares y los desarrolladores con mejores reviews. Además, se hicieron nubes de palabras para analizar la frecuencia de las mismas en las distintas columnas de los dataframes(**title**,**specs**,etc).


<p align="center">
<img src "images\juegospopulares.png" height=300>
</p>

<p align="center">
<img src="images\tags.png" height=400>
</p>

## API y RENDERIZADO
Se crearon 5 funciones para los endpoints que se consumirán en la API:

+ def **developer( *`desarrollador` : str* )**:
    `Cantidad` de items y `porcentaje` de contenido Free por año según empresa desarrolladora. 
Ejemplo de retorno:
<p align="center">
<img src="images\api1.png" height=400>
</p>

+ def **userdata( *`User_id` : str* )**:
    Devuelve `cantidad` de dinero gastado por el usuario, el `porcentaje` de recomendación en base a reviews.recommend y `cantidad de items`.

Ejemplo de retorno:
<p align="center">
<img src="images\api2.png" height=400>
</p>


+ def **UserForGenre( *`genero` : str* )**:
    Nos retorna el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.



+ def **best_developer_year( *`año` : int* )**:
   Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado.
Ejemplo de retorno: 

<p align="center">
<img src="images\api3.png" height=400>
</p>



+ def **developer_reviews_analysis( *`desarrolladora` : str* )**:
    Con el nombre de una empresa desarrolladora, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total 
    de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo. 

Ejemplo de retorno: 
<p align="center">
<img src="images\api4.png" height=400>
</p>

<br/>

## RECOMENDACIÓN

Se realizó un modelo de recomendación utilizando KNN y cosine similarity, con la librería scikit-learn. Primero se creó un data frame con las columnas relevantes a comparar y luego se evaluaron las similitudes para poder llegar a 5 items recomendados.

Ejemplo de retorno: 
<p align="center">
<img src="images\modelo.png" height=400>
</p>


## Render

Por ultimo se cargaron los documentos a un repositorio local de GITHUB, y se genera el deployment con Render, usando uvicorn y fast api.



