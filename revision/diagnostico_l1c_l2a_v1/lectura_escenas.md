# Diagnóstico de las tres escenas con los productos ópticos originales

Las 48 ventanas y las referencias permanecen fijas. Las tres escenas se eligieron para investigar discrepancias ya observadas en la evaluación externa. Los resultados describen estos casos y no estiman el comportamiento del conjunto mundial completo.

Cada fila acumula los conteos de 16 ventanas. Dentro de cada escena, las cuatro variantes comparten exactamente los píxeles válidos. El umbral aplicado es estrictamente mayor que cero para ambos índices.

| Escena | Píxeles comunes | F1 NDWI L1C | F1 NDWI L2A | F1 MNDWI L1C | F1 MNDWI L2A |
|---|---:|---:|---:|---:|---:|
| S023 | 1 048 207 | 70,25 % | 74,17 % | 85,82 % | 54,68 % |
| S025 | 1 048 576 | 84,44 % | 0,52 % | 21,58 % | 3,05 % |
| S033 | 1 043 581 | 95,79 % | 51,70 % | 86,16 % | 68,75 % |

## Escena S023

Adquisición 20190816T180921, tesela 12SVF, órbita R084. Los identificadores completos y las huellas de los archivos están en diagnostico_metadatos.json.

NDWI tuvo una mediana de 0,3127 en L1C y 0,4031 en L2A entre los 3 333 píxeles etiquetados como agua. De L1C a L2A, 1 224 pasaron de agua a fondo y 0 de fondo a agua. Ambos niveles clasificaron 43 como fondo (1,29 % del agua de referencia). En el fondo de referencia, los falsos positivos pasaron de 2 744 a 172.

MNDWI tuvo una mediana de 0,2652 en L1C y -0,1683 en L2A entre los 3 333 píxeles etiquetados como agua. De L1C a L2A, 1 740 pasaron de agua a fondo y 2 de fondo a agua. Ambos niveles clasificaron 319 como fondo (9,57 % del agua de referencia). En el fondo de referencia, los falsos positivos pasaron de 674 a 53.


## Escena S025

Adquisición 20190828T084601, tesela 37VCE, órbita R107. Los identificadores completos y las huellas de los archivos están en diagnostico_metadatos.json.

NDWI tuvo una mediana de 0,1135 en L1C y -0,3885 en L2A entre los 8 456 píxeles etiquetados como agua. De L1C a L2A, 6 272 pasaron de agua a fondo y 0 de fondo a agua. Ambos niveles clasificaron 2 162 como fondo (25,57 % del agua de referencia). En el fondo de referencia, los falsos positivos pasaron de 158 a 1.

MNDWI tuvo una mediana de 0,2530 en L1C y -0,4809 en L2A entre los 8 456 píxeles etiquetados como agua. De L1C a L2A, 7 038 pasaron de agua a fondo y 0 de fondo a agua. Ambos niveles clasificaron 1 287 como fondo (15,22 % del agua de referencia). En el fondo de referencia, los falsos positivos pasaron de 50 810 a 5.


## Escena S033

Adquisición 20190913T154901, tesela 18TXS, órbita R054. Los identificadores completos y las huellas de los archivos están en diagnostico_metadatos.json.

NDWI tuvo una mediana de 0,2380 en L1C y -0,1111 en L2A entre los 37 049 píxeles etiquetados como agua. De L1C a L2A, 21 593 pasaron de agua a fondo y 6 de fondo a agua. Ambos niveles clasificaron 2 507 como fondo (6,77 % del agua de referencia). En el fondo de referencia, los falsos positivos pasaron de 524 a 92.

MNDWI tuvo una mediana de 0,6464 en L1C y 0,0275 en L2A entre los 37 049 píxeles etiquetados como agua. De L1C a L2A, 17 331 pasaron de agua a fondo y 1 de fondo a agua. Ambos niveles clasificaron 140 como fondo (0,38 % del agua de referencia). En el fondo de referencia, los falsos positivos pasaron de 11 718 a 327.

## Alcance de la comparación

L1C proporciona reflectancia en el techo de la atmósfera. L2A proporciona reflectancia de superficie de la misma adquisición. Los TIFF de los autores se verificaron contra el tamaño y CRC de sus miembros ZIP y se identificaron con un SHA-256 local. La verificación abarca los tres miembros descargados. Las diferencias observadas incluyen el nivel de procesamiento y las decisiones de alineación y remuestreo de los productos.

Wieland et al. describen la inicialización de las máscaras mediante NDWI y correcciones iterativas por tres expertos (sección III.C, p. 1089). El artículo no especifica el valor numérico del umbral inicial. El contraste a cero no permite cuantificar las ediciones históricas ni aislar una causa de los desacuerdos. [Artículo](https://doi.org/10.1109/JSTARS.2023.3333969) · [Conjunto v1.0.1](https://zenodo.org/records/11278238)

Las distribuciones de B3, B8, B11, NDWI y MNDWI se entregan por nivel, escena y clase de referencia, junto con extremos, media, desviación poblacional y cuantiles. Los controles y las transiciones por recorte permiten reproducir las agregaciones de esta lectura.
