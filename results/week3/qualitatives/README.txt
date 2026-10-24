Resultados cualitativos de los filtros de Week 3, Task 1
=======================================================

Scope: query (query = solo consultas).

Cada carpeta contiene:
  overview.jpg  consulta original, consulta filtrada y top-5.
                Verde = imagen relevante (GT); rojo = no relevante.
  summary.txt   metricas y ranking de las 30 consultas.

La estructura es qualitatives/<filtro>/<parametro>/.
Para los filtros sin parametro (none) los archivos estan en
qualitatives/none/. Los nombres open+close y close+open se
guardan como open_close y close_open.

Consultas representativas (las mismas en todos los filtros):
  00000  None-MultiplyHue      facil
  00001  None                  la mediana corrige el top-1
  00002  None-MultiplyHue      dificil
  00008  None                  facil
  00016  UnnamedImpulseNoise   dificil
  00020  UnnamedImpulseNoise   la mediana corrige el top-1
  00022  None                  dificil
  00023  UnnamedImpulseNoise   la mediana corrige el top-1

El indice con mAP@1/mAP@5 y la ruta de cada configuracion esta en
index.csv.

Regenerar:
  python results/week3/run_qualitatives.py
  python results/week3/run_qualitatives.py --scope both
