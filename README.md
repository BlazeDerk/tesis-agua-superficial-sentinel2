# Segmentación de cuerpos de agua superficiales en Ecuador con Sentinel‑2

**Martín Alexander Peña Paladines y Brandon Estefano Castillo Ortega**

Este repositorio reúne el código y los resultados del estudio piloto en El Oro, el Golfo de Guayaquil, los lagos de Imbabura y el río Napo. El modelo delimita el agua superficial con una U-Net cuyo codificador VGG16 recibe las bandas B2, B3, B4 y B8. El estudio reunió 860 pares de imágenes y máscaras. Los filtros de calidad y la separación espacial dejaron 561 recortes de 256 × 256 píxeles distribuidos en 460 de entrenamiento, 42 de validación y 59 de prueba.

## Resultados

La comparación cubre los mismos 3 866 624 píxeles de prueba. El método NDWI clasifica como agua un píxel cuando la diferencia entre las reflectancias verde e infrarroja cercana, dividida por su suma, supera cero. Las cifras de U-Net se calcularon a partir de los conteos conservados de la corrida de 50 épocas, con un umbral de 0,5.

| Método | IoU global | F1 global | Media de F1 de las cuatro zonas |
|---|---:|---:|---:|
| U-Net con VGG16 | 88,43 % | 93,86 % | 67,53 % |
| NDWI | 80,92 % | 89,45 % | 66,03 % |

U-Net obtuvo mayor concordancia global y en las dos zonas costeras. NDWI obtuvo mayor concordancia en Imbabura y Napo. La muestra de prueba incluye un solo recorte de Imbabura y tres de Napo. El Golfo aporta 687 613 de los 908 590 píxeles de referencia clasificados como agua. Las cifras globales reflejan esa distribución. En las tres zonas incorporadas con mosaicos de Sentinel‑2 se usaron etiquetas automáticas MNDWI. La procedencia detallada de las etiquetas de El Oro permanece sin verificar. La validación con referencias humanas independientes sigue pendiente.

Cuatro recortes de prueba habían aparecido en el ensayo preliminar. El 21 de noviembre de 2026 se ejecutó de nuevo el checkpoint final sobre los 59 originales. Los conteos globales y regionales coincidieron exactamente con los de octubre. En los 55 recortes restantes, U-Net alcanzó una IoU de 87,71 % y un F1 de 93,45 %. NDWI alcanzó 79,58 % y 88,63 %, respectivamente. Imbabura quedó sin muestras en esta cohorte.

El 20 de noviembre se evaluaron además 256 ventanas de 16 escenas externas S1S2-Water con contrapartes L2A. Sobre el soporte común, U-Net obtuvo 88,49 % de IoU y 93,90 % de F1, NDWI obtuvo 91,78 % y 95,72 %, y MNDWI obtuvo 91,67 % y 95,66 %. Estas referencias se inicializaron con NDWI y fueron corregidas por tres expertos. Se incluyen la media por escena, los contrastes exploratorios con remuestreo de escenas y las estadísticas de los 460 recortes de entrenamiento. [Reproducir los análisis de revisión](revision/README.md).

## Reproducir el cálculo de NDWI

Se utilizó Python 3.12 con las versiones de NumPy y Rasterio indicadas en `requirements.txt`. La reproducción requiere acceso autorizado a los 59 pares originales. Coloque las imágenes en `Dataset_UNet_Unificado/images/` y las máscaras en `Dataset_UNet_Unificado/masks/`, con los nombres registrados en el manifiesto. Desde la raíz del repositorio ejecute

```sh
python -m pip install -r requirements.txt
python -m unittest test_baseline_ndwi
python baseline_ndwi.py --splits manifiestos/particion.csv --exposure manifiestos/exposicion.csv --manifest manifiestos/pares_prueba.csv --dataset-root Dataset_UNet_Unificado --out-dir resultados_locales
```

El script verifica cada par por SHA-256, comprueba el tipo y la cuadrícula, y calcula los resultados globales y regionales. Puede cotejar el archivo generado `ndwi_resumen.csv` con el de `resultados/`. El CSV por recorte contiene los conteos de verdaderos positivos, falsos positivos, falsos negativos y verdaderos negativos. El JSON identifica los insumos por nombre y huella. Los tiempos de ejecución del JSON corresponden al cálculo de NDWI.

## Contenido y acceso

- `manifiestos/` contiene la partición final de 561 recortes, los identificadores y hashes de los 59 pares de prueba y las cuatro exposiciones documentadas. Son extractos del registro histórico con las columnas necesarias para reproducir el cálculo.
- `resultados/` contiene el NDWI por recorte, su resumen y metadatos, la comparación regional con U-Net y los conteos individuales disponibles del modelo. Las celdas vacías de U-Net identifican conteos que faltan.
- `metodo_original/` conserva los scripts de Earth Engine, preparación de parches, entrenamiento y reanudación, junto con la configuración y el historial. La reproducción del entrenamiento original requiere su entorno, los datos completos y, para la reanudación, el checkpoint intermedio.

## Consultar el procesamiento y el modelo

El script `metodo_original/gee_v1_ejecutado.js` se ejecuta en el editor de Google Earth Engine con un proyecto habilitado. Exporta los mosaicos de las tres zonas incorporadas. Para preparar el dataset completo se necesitan también las parejas históricas de El Oro. El script de recorte permite indicar las carpetas con `--inputs`, `--legacy` y `--output`. El script de entrenamiento acepta `--dataset`, `--manifest` y `--output`. El archivo de configuración documenta los parámetros originales. Se utilizó TensorFlow 2.20.0 con Keras 3.15.1. Las dependencias de entrenamiento están en `requirements-entrenamiento.txt`.

La reanudación documentada en `reanudar_final_20261005.py` busca los archivos conservados en la estructura original del proyecto y comprueba sus hashes. Para consultarla como registro histórico basta con leer el script. Los comandos de reproducción anteriores ejecutan la comparación NDWI.

El acceso a los GeoTIFF y pesos se gestiona por separado. La autorización para redistribuir los datos antiguos de El Oro está pendiente de documentación.
