# Reproducción de la revisión del 21 de noviembre de 2026

El cálculo por escena usa las tablas de conteos incluidas y se puede ejecutar en un checkout limpio. El evaluador de píxeles requiere los GeoTIFF correspondientes y el checkpoint final de la época 50. Los resultados de referencia están en las carpetas de esta revisión. Los comandos siguientes se ejecutan desde la raíz del repositorio.

## Estadísticas externas

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s revision/analisis_local -p "test_*.py" -v
python revision/analisis_local/analizar_escenas.py
```

El programa lee los conteos de 16 escenas en `revision/insumos_externos/conteos` y crea `revision/estadistica_recalculada` en un destino nuevo. El bootstrap utiliza 20 000 remuestreos pareados de escenas con semilla 20261120. El efecto global por píxeles y el promedio que asigna el mismo peso a cada escena responden a preguntas distintas. La incertidumbre es exploratoria y está condicionada a estas escenas y a un solo checkpoint.

## Inferencia histórica y distribución de entradas

La columna `name` identifica cada imagen y máscara, que deben estar bajo `images/` y `masks/` dentro del directorio indicado. Sustituya las rutas de ejemplo por las ubicaciones autorizadas del dataset y del checkpoint en su equipo. Se utilizó Python 3.12.14 en CPU con TensorFlow 2.20.0, Keras 3.15.1, NumPy 2.2.6 y Rasterio 1.5.1.

```sh
python -m pip install -r revision/requirements-revision.txt
python revision/reevaluar_historicos.py --dataset-root RUTA_DATASET --splits manifiestos/particion.csv --manifest manifiestos/pares_prueba.csv --exposure manifiestos/exposicion.csv --validate-only --out-dir control59_local
python revision/reevaluar_historicos.py --dataset-root RUTA_DATASET --splits manifiestos/particion.csv --manifest manifiestos/pares_prueba.csv --exposure manifiestos/exposicion.csv --checkpoint RUTA_MODELO/model_best.keras --out-dir historicos_local --batch-size 4
python revision/estadisticos_train_congelado.py --splits manifiestos/particion.csv --images-dir RUTA_DATASET/images --out train_local.json
python revision/comparar_bandas.py --train train_local.json --external revision/analisis_local/estadisticos_bandas_externa.json --out comparacion_train_local.csv
```

El checkpoint pesa 199 255 390 bytes y su SHA-256 es `9e28f92b557bdd2e9a2e7b068a6622254ddbf80080926b8c004368ad5683dda4`. Los scripts de publicación aceptan tanto los hashes íntegros de los registros originales como los hashes fijos de las proyecciones portátiles de esos registros. Se conservan todos los controles de identificación, pares de hashes, soporte y cuadrículas. La corrida documentada en `fuentes_y_ejecuciones.json` empleó los registros originales. La correspondencia de hashes de los registros y sus proyecciones está en `insumos_historicos/proyecciones_publicas.json`.

En `resultados_historicos/` se encuentran las 118 filas por recorte y método, el resumen de ambas cohortes, los contrastes U-Net menos NDWI y el control de octubre. Las diferencias de métricas se expresan en puntos porcentuales. En la tabla del subconjunto de 55, Imbabura figura sin muestras. El análisis usa una nueva inferencia del modelo original. Las tablas anteriores de `../resultados/` conservan el estado histórico de recuperación anterior a esta inferencia.

En `bandas_train/` se indican medias, desviaciones poblacionales, extremos y tamaños por banda, globales y por zona. El cálculo usa los 460 Train antes del aumento y de la capa afín. La tabla externa distingue el soporte completo tras el relleno, los valores finitos y los píxeles comunes de evaluación. Las diferencias de media y ambas orientaciones de la razón de desviaciones están identificadas en las tablas.

## Píxeles y distribución de datos

La tabla `inventario_probabilidades.csv` contiene los nombres, tamaños y SHA-256 de las 59 probabilidades históricas y las 256 externas. El equipo conserva los TIFF en las dos entregas provistas al revisor. El paquete externo se llama `resultados_externos_v1.zip` y la entrega de revisión contiene `resultados_hist59_nueva_inferencia_v1/probabilidades/`. Los hashes permiten cotejar cada archivo antes del análisis. La disponibilidad pública de los binarios está pendiente de un depósito autorizado.

Para la evaluación de las ventanas externas, utilice una copia autorizada de `Paquete_evaluacion_externa_S1S2Water` que contenga `manifest_s1s2.csv`, `datos/`, `fuentes/`, `plan_congelado.json` y `estado_adquisicion.json`. El manifiesto, el plan y las tablas conservadas están incluidos en `insumos_externos/`. Los archivos L2A preparados ya son reflectancias físicas en B2, B3, B4, B8 y B11. El modelo recibe las cuatro primeras bandas y B11 interviene en MNDWI.

```sh
python revision/inferir_y_evaluar.py --manifest RUTA_PAQUETE_EXTERNO/manifest_s1s2.csv --validate-only --out-dir control_externo_local
python revision/inferir_y_evaluar.py --manifest RUTA_PAQUETE_EXTERNO/manifest_s1s2.csv --checkpoint RUTA_MODELO/model_best.keras --out-dir externo_local --batch-size 4
python revision/graficar_resultados.py --results externo_local
```

## Contraste entre L1C y L2A

El plan se limita a los tres TIFF ópticos L1C originales de S023, S025 y S033. `obtener_l1c_original.py` lee la tabla central conservada del ZIP de los autores. El diagnóstico conserva las 16 posiciones por escena y compara los índices a cero sobre el soporte compartido. Los índices de los autores se utilizaron como punto de partida de las referencias y el umbral original no está documentado.

```sh
python revision/obtener_l1c_original.py --index-root revision/insumos_externos/fuentes
python revision/obtener_l1c_original.py --index-root revision/insumos_externos/fuentes --out-dir l1c_original --max-gb 6 --download
python revision/diagnosticar_l1c_l2a.py --scene-ids 23,25,33 --l1c-root l1c_original --external-root RUTA_PAQUETE_EXTERNO --manifest revision/insumos_externos/manifest_s1s2.csv --plan revision/insumos_externos/plan_congelado.json --archive-index-root revision/insumos_externos/fuentes --out-dir contraste_local
```

## Procedencia y derechos

Las evidencias documentales sobre el ensayo técnico y los 555 pares de El Oro están bajo `decisiones_y_procedencia/`. La carpeta `catalogo_candidatos/` registra una consulta del 21 de noviembre de 2026 a Earth Search sobre los sectores y el periodo del estudio. Se obtuvieron 1 077 productos candidatos para el Golfo, 285 para Imbabura y 364 para Napo. Los conteos originales del cruce de GEE fueron 1 414, 417 y 601. El campo destinado al identificador histórico de GEE se conserva vacío cuando falta evidencia del registro original. El comando `python revision/reconstruir_candidatos.py` guarda una nueva consulta en un directorio separado. Un catálogo reprocesado puede cambiar con el tiempo.

Las imágenes y máscaras antiguas de El Oro y los pesos del modelo se conservan en el equipo del proyecto. La procedencia y los permisos de redistribución de El Oro requieren documentación adicional. La licencia del código del proyecto queda pendiente de declaración de sus titulares.

S1S2-Water v1.0.1, de Wieland y colaboradores, tiene licencia CC BY 4.0. Las ventanas externas usan referencias de ese conjunto y óptica Sentinel-2 L2A de la misma adquisición. Se aplicaron vecino más próximo para las cuatro bandas de 10 m e interpolación bilineal para B11. Las condiciones de Copernicus para los productos ópticos se documentan por separado. [Atribución de las fuentes externas](insumos_externos/ATRIBUCION.md) · [S1S2-Water en Zenodo](https://zenodo.org/records/11278238).
