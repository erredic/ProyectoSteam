from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional

app = FastAPI()

#presentacion de la api
@app.get('/')
def mensaje():
    return "Hola! Esta API está diseñada para resolver dudas sobre Steam."

#Consulta #1 Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora.
@app.get('/developer/{desarrollador}')
def developer(desarrollador: str):
    # Ruta del archivo Parquet
    file_path = r"C:\Users\mateo\OneDrive\Escritorio\data science\labs\PROYECTOSTEAAM\PARQUET\SteamGames.parquet"
    
    # Cargar el archivo Parquet
    try:
        steam_games_df = pd.read_parquet(file_path)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Archivo Parquet no encontrado.")
    
    # Verificar si el desarrollador existe en el conjunto de datos
    if desarrollador not in steam_games_df['developer'].unique():
        raise HTTPException(status_code=404, detail="Desarrollador no encontrado. Revisa la ortografía.")
    
    # Filtrar el dataframe por el desarrollador especificado
    df_developer = steam_games_df[steam_games_df['developer'] == desarrollador].copy()

    # Convertir la columna 'release_date' a tipo datetime usando .loc
    df_developer.loc[:, 'release_date'] = pd.to_datetime(df_developer['release_date'])

    # Agrupar por año y contar la cantidad de juegos lanzados por año
    games_by_year = df_developer.groupby(df_developer['release_date'].dt.year)['title'].count()

    # Calcular la cantidad de juegos con contenido gratuito por año
    free_games_by_year = df_developer[df_developer['price'] == 'Free'].groupby(df_developer['release_date'].dt.year)['title'].count()

    # Calcular el porcentaje de contenido gratuito por año
    percentage_free_by_year = (free_games_by_year / games_by_year) * 100

    # Crear un DataFrame para almacenar los resultados
    results_df = pd.DataFrame({
        'Año': games_by_year.index,
        'Cantidad de Items': games_by_year.values,
        'Contenido Free': percentage_free_by_year.fillna(0).values.astype(int)
    })

    # Devolver los resultados
    return results_df.to_dict(orient='records')

#Dconsulta #2 ebe devolver cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.
@app.get("/user/{user_id}")
def userdata(user_id: str):
    # Cargar los archivos Parquet
    user_items_df = pd.read_parquet(r"C:\Users\mateo\OneDrive\Escritorio\data science\labs\PROYECTOSTEAAM\PARQUET\UsersItems.parquet")
    user_reviews_df = pd.read_parquet(r"C:\Users\mateo\OneDrive\Escritorio\data science\labs\PROYECTOSTEAAM\PARQUET\UserReviews.parquet")
    steam_games_df = pd.read_parquet(r"C:\Users\mateo\OneDrive\Escritorio\data science\labs\PROYECTOSTEAAM\PARQUET\SteamGames.parquet")

    # Verificar si el ID de usuario existe en ambos DataFrames
    if user_id not in user_items_df['user_id'].values:
        raise HTTPException(status_code=404, detail="Usuario no encontrado.")

    # Filtrar el DataFrame de items por el usuario dado usando .loc
    user_items_filtered = user_items_df.loc[user_items_df['user_id'] == user_id].copy()

    # Asegurarse de que 'item_id' en ambos DataFrames sea del mismo tipo
    user_items_filtered.loc[:, 'item_id'] = user_items_filtered['item_id'].astype(int)
    steam_games_df.loc[:, 'item_id'] = steam_games_df['item_id'].astype(int)

    # Obtener los precios de los juegos comprados por el usuario
    precios_juegos = user_items_filtered.merge(
        steam_games_df[['item_id', 'price']], 
        left_on='item_id', 
        right_on='item_id',
        how='left'
    )['price']

    # Convertir los precios a números
    precios_juegos = pd.to_numeric(precios_juegos, errors='coerce')

    # Calcular el total gastado por el usuario
    total_gastado = precios_juegos.sum()

    # Filtrar el DataFrame de revisiones por el usuario dado usando .loc
    user_reviews_filtered = user_reviews_df.loc[user_reviews_df['user_id'] == user_id].copy()

    # Calcular el porcentaje de recomendación
    if len(user_reviews_filtered) > 0:
        porcentaje_recomendacion = (user_reviews_filtered['recommend'].sum() / len(user_reviews_filtered)) * 100
        mensaje_recomendacion = f"{porcentaje_recomendacion:.2f}% de recomendación"
    else:
        mensaje_recomendacion = "Este jugador nunca ha recomendado"

    # Calcular la cantidad de items del usuario
    cantidad_items = len(user_items_filtered)

    # Crear el diccionario de retorno
    retorno = {
        "Usuario": user_id,
        "Dinero gastado": f"{total_gastado:.2f} USD",
        "% de recomendación": mensaje_recomendacion,
        "Cantidad de items": cantidad_items
    }

    return retorno

#consulta #3 Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.

@app.get("/consulta/{genero}")
async def consultar_informacion(genero: str):
    try:
        # Cargar los archivos CSV en los pandas DataFrames
        DFSteamGames = pd.read_csv(r"C:\Users\mateo\OneDrive\Escritorio\data science\labs\PROYECTOSTEAAM\CSV\SteamGames.csv")
        DFUserItems = pd.read_csv(r"C:\Users\mateo\OneDrive\Escritorio\data science\labs\PROYECTOSTEAAM\CSV\UsersItems.csv")

        # Verificar que las columnas 'item_id' existan en ambos DataFrames
        if 'item_id' not in DFSteamGames.columns or 'item_id' not in DFUserItems.columns:
            raise HTTPException(status_code=400, detail="La columna 'item_id' no se encuentra en uno de los archivos CSV.")

        # Eliminar filas con valores NaN en las columnas 'genres' y 'item_id'
        DFSteamGames = DFSteamGames.dropna(subset=['genres', 'item_id'])
        DFUserItems = DFUserItems.dropna(subset=['item_id', 'playtime_forever'])

        # Filtrar los juegos por el género dado
        juegos_genero = DFSteamGames[DFSteamGames['genres'].str.contains(genero, case=False, na=False)]

        # Verificar si hay juegos asociados al género dado
        if juegos_genero.empty:
            raise HTTPException(status_code=404, detail=f"No hay juegos asociados al género '{genero}'.")

        # Unir los DataFrames para obtener las horas jugadas por usuario
        horas_jugadas_por_usuario = pd.merge(juegos_genero, DFUserItems, on='item_id')

        # Eliminar filas con valores NaN en la columna 'playtime_forever' después de la fusión
        horas_jugadas_por_usuario = horas_jugadas_por_usuario.dropna(subset=['playtime_forever'])

        # Sumar las horas jugadas por usuario
        horas_por_usuario = horas_jugadas_por_usuario.groupby('user_id')['playtime_forever'].sum().reset_index()

        # Verificar si hay datos para el género dado
        if horas_por_usuario.empty:
            return {"mensaje": f"No hay datos disponibles para el género '{genero}'."}

        # Obtener el usuario con más horas jugadas
        usuario_mas_horas = horas_por_usuario.loc[horas_por_usuario['playtime_forever'].idxmax()]

        # Calcular las horas jugadas por año de lanzamiento
        juegos_con_anio = pd.merge(DFSteamGames, DFUserItems, on='item_id')
        # Eliminar filas con valores NaN en la columna 'playtime_forever' antes de calcular la suma por año
        juegos_con_anio = juegos_con_anio.dropna(subset=['playtime_forever'])
        horas_por_anio = juegos_con_anio.groupby('release_date')['playtime_forever'].sum().reset_index()

        return {
            "Usuario_mas_horas": usuario_mas_horas.to_dict(),
            "Horas_jugadas_por_anio": horas_por_anio.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# consulta #4 Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos)

@app.get("/best_developer_year/{year}")
def best_developer_year(year: int):
    # Cargar los DataFrames
    steam_games_df = pd.read_parquet(r"C:\Users\mateo\OneDrive\Escritorio\data science\labs\PROYECTOSTEAAM\PARQUET\SteamGames.parquet")
    user_reviews_df = pd.read_parquet(r"C:\Users\mateo\OneDrive\Escritorio\data science\labs\PROYECTOSTEAAM\PARQUET\UserReviews.parquet")

    # Eliminar filas con valores faltantes en la columna 'release_date'
    steam_games_df = steam_games_df.dropna(subset=['release_date'])

    # Convertir la columna 'release_date' a tipo datetime
    steam_games_df['release_date'] = pd.to_datetime(steam_games_df['release_date'], errors='coerce')

    # Filtrar los juegos por el año proporcionado
    juegos_year = steam_games_df[steam_games_df['release_date'].dt.year == year]

    # Si no hay juegos para el año dado, devolver un error HTTP 404
    if juegos_year.empty:
        raise HTTPException(status_code=404, detail="No se encontraron juegos para el año proporcionado.")

    # Obtener las revisiones para los juegos del año dado
    reviews_year = user_reviews_df[user_reviews_df['item_id'].isin(juegos_year['item_id'])]

    # Filtrar las revisiones recomendadas
    reviews_recomendadas = reviews_year[(reviews_year['recommend'] == True) & (reviews_year['sentiment_analysis'] > 0)]

    # Unir las revisiones con la información del juego (desarrollador)
    reviews_juegos_year = reviews_recomendadas.merge(steam_games_df, left_on='item_id', right_on='item_id', how='inner')

    # Contar la cantidad de revisiones por desarrollador
    desarrolladores_recomendados = reviews_juegos_year.groupby('developer')['recommend'].count()

    # Obtener el top 3 de desarrolladores con más revisiones recomendadas
    top_desarrolladores = desarrolladores_recomendados.nlargest(3)

    # Formatear los resultados en el formato requerido
    resultados = [{"Puesto {}: {}".format(idx+1, desarrollador): juegos_recomendados} for idx, (desarrollador, juegos_recomendados) in enumerate(top_desarrolladores.items())]

    return resultados


# Consulta #5 Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo.

@app.get("/developer_reviews_analysis/{desarrolladora}")
def developer_reviews_analysis(desarrolladora: str):
    # Cargar los DataFrames
    steam_games_df = pd.read_parquet(r"C:\Users\mateo\OneDrive\Escritorio\data science\labs\PROYECTOSTEAAM\PARQUET\SteamGames.parquet")
    user_reviews_df = pd.read_parquet(r"C:\Users\mateo\OneDrive\Escritorio\data science\labs\PROYECTOSTEAAM\PARQUET\UserReviews.parquet")

    # Filtrar las revisiones para la desarrolladora especificada
    reviews_desarrolladora = user_reviews_df[user_reviews_df['item_id'].isin(steam_games_df[steam_games_df['developer'] == desarrolladora]['item_id'])]

    # Filtrar las revisiones con análisis de sentimiento positivo o negativo
    reviews_positivas = reviews_desarrolladora[reviews_desarrolladora['sentiment_analysis'] > 0]
    reviews_negativas = reviews_desarrolladora[reviews_desarrolladora['sentiment_analysis'] < 0]

    # Contar la cantidad de revisiones positivas y negativas
    positivas_count = reviews_positivas.shape[0]
    negativas_count = reviews_negativas.shape[0]

    # Construir el diccionario de retorno
    retorno = {desarrolladora: {"Positive": positivas_count, "Negative": negativas_count}}

    return retorno


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

