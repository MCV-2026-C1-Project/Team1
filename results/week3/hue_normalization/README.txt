Normalizacion de tono en HSV con retrieval en LAB
=================================================

Referencias: C:\Users\andre\Desktop\MCV\C1\Team1\results\week3\color_reference\museum_hue_references.npz
Codigo de referencias: src/color_reference/
Museo: C:\Users\andre\Desktop\MCV\C1\Team1\data\BBDD

Metodo
------
1. Se construye un histograma de tono por imagen de BBDD, ponderado
   por saturacion y ignorando pixeles con S <= umbral.
2. Para cada candidato del museo se estima un factor global de escala
   de tono que alinea el histograma de la query con su referencia.
3. La correccion se aplica solo al canal H de la query (HSV), dejando
   S y V intactos, y se vuelve a RGB.
4. El descriptor sigue siendo 'Lab blocks 1D' y la distancia L1; el
   ranking final mezcla la distancia LAB original y la normalizada,
   ponderando cada candidato por la confianza del alineamiento de tono.

No se usa la etiqueta de augmentacion en ningun paso.

Parametros
----------
  top_k: 5
  sat_threshold: 40
  scale_min: 0.75
  scale_max: 1.3
  scale_step: 0.005
  descriptor_step: 0.01
  regularization: 0.1
  alpha: 1.0
  temperature: 1.0
  blocks: 25
  bins: 32

Resultados
----------
  all     baseline         mAP@1=0.6667 mAP@5=0.7511
  all     global_reference mAP@1=0.6000 mAP@5=0.6750
  all     hue_reference    mAP@1=0.7333 mAP@5=0.7694
  hue     baseline         mAP@1=0.7143 mAP@5=0.7857
  hue     global_reference mAP@1=0.6429 mAP@5=0.7143
  hue     hue_reference    mAP@1=0.7857 mAP@5=0.8095
  clean   baseline         mAP@1=0.6667 mAP@5=0.7778
  clean   global_reference mAP@1=0.6667 mAP@5=0.7222
  clean   hue_reference    mAP@1=0.7778 mAP@5=0.8333
  impulse baseline         mAP@1=0.5714 mAP@5=0.6476
  impulse global_reference mAP@1=0.4286 mAP@5=0.5357
  impulse hue_reference    mAP@1=0.5714 mAP@5=0.6071

Notas
-----
- La referencia global (media de BBDD) se evalua como control; no usa
  una referencia distinta por candidato.
- La mejora se concentra en las queries None-MultiplyHue; las queries
  con ruido impulsivo no se corrigen con este metodo.
- El ranking de cada query y el factor estimado para su GT estan en
  per_query.csv.
