Resultados cualitativos con MSE y ranking L1
===============================================

Carpeta nueva; los resultados anteriores no se han modificado.

Filtros: average, gaussian, wavelet_lowpass
Muestras: 00016.jpg (UnnamedImpulseNoise)
Kernel fijo: 9x9 para todos los filtros con kernel.
Wavelet: db2, nivel 3.

Cada filtro tiene una imagen por muestra seleccionada:
  <filtro>/00016.jpg

En cada imagen:
  izquierda = consulta original con ruido
  centro    = flecha con el MSE entre original y filtrada
  derecha   = consulta tras aplicar el filtro
  pie       = ID de la imagen GT y su posicion en el ranking L1

MSE: media de (original - filtrada)^2 sobre los tres canales RGB,
con valores de pixel en 0-255.
Ranking: 1-based, sobre las 287 imagenes de BBDD, usando L1.

El resumen numerico esta en index.csv.
