Resultados cualitativos con MSE y ranking L1
===============================================

Carpeta nueva; los resultados de qualitatives/ no se han modificado.

Muestras: 00018.jpg y 00020.jpg (ambas UnnamedImpulseNoise).
Kernel fijo: 9x9 para todos los filtros con kernel.
Wavelet: db2, nivel 3.

Cada filtro tiene dos imagenes, una por muestra:
  <filtro>/00018.jpg
  <filtro>/00020.jpg
Los filtros open+close y close+open se guardan como open_close y
close_open.

En cada imagen:
  izquierda = consulta original con ruido
  centro    = flecha con el MSE entre original y filtrada
  derecha   = consulta tras aplicar el filtro
  pie       = ID de la imagen GT y su posicion en el ranking L1

MSE: media de (original - filtrada)^2 sobre los tres canales RGB,
con valores de pixel en 0-255.
Ranking: 1-based, sobre las 287 imagenes de BBDD, usando L1.

El resumen numerico esta en index.csv.
