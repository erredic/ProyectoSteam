
# Proyecto Steam

este proyecto va a llevar a cabo una limpieza y investigacion de los datos de tres archivos json.gz los cuales podremos encontrar en la carpeta de "datos base". A lo largo del Readme vamos encontrar una descripcion del contenido de las carpetas.

## Introduccion a las carpetas

La primera carpeta que se recomienda ver es "datos base" en esta carpeta donde encontraremos los archivos json, el diccionario datos y  junto a esto  un archivo llamado "descomprimir datos.ipynb" en este archivo se descomprimen y se transforman los datos a json los cuales se guardan en esa carpeta.

## ETL

En esta fase usamos los archivos .json y los hacemos legibles, a la par creamos archivos csv y parquet con base a estos. Estos archivos se pueden encontrar en las carpetas parquet y csv.

### Cleandata

Es una carpeta que aunque no se encuentre dentro de la carpeta de ETL es un archivo que se realizo para limpiar datos vacios.

## API

esto se divide en dos carpetas principalmente:


### Consultas API

En esta carpeta vamos a poder ver las consultas pedidas en archivos de jupyternotebook a la par que archivos de csv y parquet.

### API 

En esta carpeta vamos a poder encontrar la API, siendo su archivo mas importante el main

# Analisis EDA

Aca vamos a moldear un poco los datos con el fin de conseguir informacion adicional o de interes de estos, principalmente enfocados en ver que tipo de juegos son mas vendidos y populares para la comunidad.

# Machine learning - recomendacion de videojuegos

Esto lo vamos a ver hospedado en la carpeta "ML_RECOMENCACION DE VIDEOJUEGOS" donde es una funcion que entregandole el ID del videojuego nos va a recomendar 5 videojuegos similares con el fin de tener recomendaciones de estos.

# requirements

Los siguientes requirements son para la carpeta sin contar el API, para el API los archivos se van a poder ver en su carpeta.
pandas
sklearn
matplotlib
IPython.display
numpy
scipy.sparse
typing
gzip
shutil



#conclusiones 

-Se realiza limpieza de datos.
-Se realiza orden de los datos.
-Se creo una API que resuelve consultas de multiples tipos.
-Se analizaron los datos en un EDA con el fin de poder obtener datos extra.
-se obtuvo un sistema de machine learning el cual funciona con similitud del coseno para recomendar videojuegos.
