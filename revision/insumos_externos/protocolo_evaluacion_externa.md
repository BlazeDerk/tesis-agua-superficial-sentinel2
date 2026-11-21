# Protocolo de evaluación externa con referencias S1S2-Water y Sentinel-2 L2A

Preparación: 18 de noviembre de 2026. Este documento establece decisiones para una evaluación externa con referencias publicadas revisadas por expertos y contrapartes Sentinel-2 L2A de las mismas adquisiciones. Su existencia no significa que la evaluación ya se haya ejecutado. La selección definitiva se registrará en un manifiesto antes de consultar predicciones o métricas.

## Pregunta y alcance

La pregunta es si el checkpoint del piloto conserva concordancia con una referencia externa revisada por expertos y cómo se compara con NDWI y MNDWI bajo las mismas condiciones de evaluación. Se utilizarán las anotaciones publicadas de S1S2-Water, sin pedir a los autores de este artículo que interpreten o dibujen nuevas máscaras.

Esta prueba evalúa transferencia a otro conjunto de imágenes. Sus resultados se presentarán separados de los 59 recortes históricos de Ecuador. No se sumarán sus píxeles ni sus métricas a los del piloto original, y no se interpretarán como validación específica de El Oro, el Golfo, Imbabura o Napo.

El artículo primario de S1S2-Water, página 1089 (página 6 del PDF), describe un proceso semiautomático: una delimitación inicial por umbral NDWI se refina mediante múltiples iteraciones de comprobación y corrección por tres expertos independientes en interpretación de imágenes. Las máscaras se ajustan a la imagen específica de cada sensor. Por ello, la referencia no es íntegramente manual desde cero ni completamente independiente del NDWI. Se declarará un posible sesgo de anclaje al compararla con ese índice. Aporta control experto externo y un origen distinto del MNDWI usado en las etiquetas nuevas del piloto.

No debe denominarse «anotación ciega realizada por nuestros revisores» ni «verdad de campo». Su método, calidad y posibles limitaciones se atribuirán a los autores del conjunto. La combinación de referencias S1S2-Water con L2A se identificará como una adaptación del conjunto; no se presentará como ejecución exacta del benchmark original, cuya óptica es L1C.

## Datos y selección antes de calcular resultados

Fuentes a conservar:

- [Repositorio de los autores de S1S2-Water](https://github.com/MWieland/s1s2_water).
- [Depósito de S1S2-Water, versión 1.0.1](https://zenodo.org/records/11278238).
- [Artículo del conjunto](https://doi.org/10.1109/JSTARS.2023.3333969).

Se conservarán versión, URL de cada archivo, leyendas de clases, identificadores originales, metadatos y condiciones de reutilización. Si se utiliza una distribución de parches de un tercero, se documentarán su transformación, versión y correspondencia con las escenas originales. No se supondrá esa equivalencia solo por compartir el nombre del conjunto.

La documentación de los autores describe la óptica Sentinel-2 como seis bandas UInt16 en el orden azul, verde, rojo, infrarrojo cercano, SWIR1 y SWIR2, correspondientes a B2, B3, B4, B8, B11 y B12. En esta ruta se adquirirán `MSK` y `VALID` del depósito, y las contrapartes L2A de las mismas adquisiciones desde un catálogo público trazable. `MSK` usa 0=fondo y 1=agua; `VALID` usa 0=no válido y 1=válido, y excluye nubes, sombras y ausencia de datos. El depósito v1.0.1 declara licencia CC BY 4.0; se conservarán la atribución y la versión de la licencia junto al subconjunto utilizado.

Los 65 metadatos de los autores contienen 33 escenas de entrenamiento, 16 de validación y 16 de prueba. Se reservan las escenas del Test oficial con identificadores numéricos 23, 25, 29, 31, 33, 35, 47, 53, 57, 75, 77, 78, 80, 82, 88 y 89. Se intentará adquirir las referencias y contrapartes L2A de las dieciséis. Los productos ausentes o cuya identidad no pueda verificarse se registrarán como faltantes, conservando la cohorte prevista y la cohorte finalmente disponible.

Dentro de cada escena se enumerarán todas las ventanas completas no solapadas de 256 × 256 píxeles, ancladas en la fila y columna cero de la cuadrícula de la referencia. Se ordenarán por el hexadecimal SHA-256 de la cadena ASCII literal `s1s2water-l2a-v1|{scene_id}|r{row_off}|c{col_off}`, sustituyendo las llaves por los identificadores y offsets correspondientes. Se elegirán las primeras dieciséis ventanas por escena antes de consultar las predicciones. La muestra prevista alcanza 256 ventanas si se dispone de las dieciséis escenas. Es un muestreo de ventanas del Test oficial, no una evaluación de todos los píxeles del Test.

No se filtrarán candidatos por cantidad de agua ni se sustituirán ventanas después de observar métricas. La validez y el soporte efectivo se informarán por separado: una ventana seleccionada que quede sin píxeles evaluables permanecerá en el inventario con ese estado. El plan guardará los identificadores y coordenadas antes de calcular resultados.

La selección no dependerá del área predicha por U-Net, NDWI o MNDWI, de los valores de IoU/F1 ni de inspeccionar ejemplos para escoger los más favorables. Deben poder entrar escenas y ventanas sin agua. Las exclusiones por formato, archivos ausentes o ausencia total de soporte válido se registrarán con su motivo.

Se contrastarán la geometría y las fechas disponibles con las huellas históricas y los ejemplos externos usados durante el desarrollo. Las escenas que compartan adquisición o ubicación se agruparán conservadoramente. Usar una fuente pública distinta no prueba por sí solo independencia espacial o temporal.

## Correspondencia entre la referencia y L2A

Para cada escena se extraerán `s2_srcids` y `date_s2` de los metadatos originales. La identidad se comprobará con el instante de toma, la plataforma Sentinel-2, la tesela MGRS y, cuando estén disponibles, órbita y gránulo. El campo genérico `properties.datetime` de los metadatos del conjunto no debe sustituir esos identificadores: existen registros cuyo valor es 2020-01-01 y cuya adquisición Sentinel-2 indicada en `s2_srcids` corresponde a otra fecha.

La búsqueda de L2A debe corresponder a la misma adquisición, no a una imagen próxima en el tiempo o simplemente al mismo mes. Se conservarán los identificadores originales y del proveedor L2A, el instante de adquisición, las URLs, la escala y el desplazamiento radiométricos y la transformación geográfica. Cuando un catálogo no conserve un producto coincidente, se registrará el faltante; no se sustituirá silenciosamente por otra fecha.

La cuadrícula de evaluación será la de `MSK` y `VALID`. Se comprobarán CRS, origen, tamaño de píxel, orientación, extensión y dimensiones. Las bandas L2A se alinearán explícitamente con ella mediante WarpedVRT: vecino próximo para B2, B3, B4 y B8 de 10 m y remuestreo bilineal para B11 de 20 m a la cuadrícula de 10 m. Las máscaras categóricas conservarán sus valores y cuadrícula, sin remuestreo. La identidad temporal no elimina la necesidad de verificar la alineación espacial. Las dimensiones de las referencias no son uniformes: los datos verificados incluyen 10980 × 10980, 10981 × 10981 y 10982 × 10981 píxeles, por lo que se utilizará la geometría real de cada archivo.

## Modelo fijo y compatibilidad radiométrica

Se conservará el checkpoint seleccionado en la época 50 del piloto, con SHA-256:

`9e28f92b557bdd2e9a2e7b068a6622254ddbf80080926b8c004368ad5683dda4`

La copia local preparada del repositorio declara que no incluye los pesos. El archivo de modelo debe localizarse o proporcionarse y comprobarse contra ese hash antes de generar resultados atribuibles al checkpoint original. Si se necesita reconstruir la arquitectura, debe conservarse exactamente la capa personalizada, el orden de las bandas y los pesos; no basta con construir una U-Net parecida.

Las cuatro entradas son B2, B3, B4 y B8, en ese orden. B11 se reserva para MNDWI y no se añade como quinta entrada a la red. El tensor de entrada tiene tamaño 256 × 256 × 4. Deben verificarse nombres, tipos, escala y ausencia de inversión de canales.

El conjunto publicado utiliza Sentinel-2 L1C, con reflectancia en el techo de la atmósfera (TOA), mientras que el piloto se entrenó con reflectancia de superficie L2A. Para acercar la entrada al producto utilizado en entrenamiento y evitar descargar una imagen L1C completa por escena, esta ruta recupera L2A de la misma adquisición y conserva la referencia publicada. El producto así formado es una adaptación S1S2-Water + L2A. Sigue existiendo un cambio de dominio geográfico y de adquisición, y la coincidencia numérica de escalas no demostraría por sí sola equivalencia de los productos.

Se aplicarán escala y desplazamiento de cada asset L2A una sola vez, según sus metadatos y nivel de procesamiento, sin suponer automáticamente que todos los proveedores entregan DN equivalentes a los exportados desde GEE. Al suministrar reflectancia a la red se mantendrá su capa interna VGG4Preprocess; no se dividirá otra vez por 10 000 un valor que ya sea reflectancia. Si un envoltorio de inferencia espera DN, la conversión a su contrato se documentará y verificará antes de ejecutar. No se eliminará ni duplicará la transformación interna de la red.

Una eventual evaluación directa sobre el L1C publicado tendrá otro protocolo/versionado y declarará el cambio TOA frente a L2A. No se mezclará con esta ejecución ni se considerará intercambiable con ella.

## Preparación espacial e inferencia

La referencia, las bandas y la máscara de validez deben compartir tamaño, transformación geográfica y cuadrícula. Una máscara categórica no se interpolará por métodos continuos. Se registrará cualquier remuestreo aplicado a B11, cuya resolución nativa difiere de las bandas de 10 m.

La inferencia se ejecutará sobre las dieciséis ventanas completas por escena fijadas por el plan. La cuadrícula será de 256 × 256, anclada en la fila y columna cero de la referencia, sin solapes. Los restos de borde que no forman una ventana completa quedan fuera del universo de muestreo; no se rellenan ni se presentan como ventanas evaluadas. Se conservarán la posición y el identificador de la escena original de cada ventana.

No se ajustará el tamaño de ventana, se escogerán umbrales, se reentrenará, se cambiarán pesos o se seleccionará otro checkpoint en función del resultado externo. Una corrección de un error de implementación debe producir una ejecución nueva, con motivo y versión documentados.

## Métodos y soporte común

Se aplicarán umbrales fijos:

| Método | Entrada | Regla de agua |
|---|---|---|
| U-Net/VGG16 | B2, B3, B4 y B8 | Probabilidad ≥ 0,5 |
| NDWI | (B3 − B8)/(B3 + B8) | Índice > 0 |
| MNDWI | (B3 − B11)/(B3 + B11) | Índice > 0 |

Los índices se calcularán con tipo flotante antes de restar, sin desbordamiento de enteros. Se exigirá denominador estrictamente positivo para ambos. El índice exactamente cero se clasificará como fondo.

El soporte de comparación será la intersección de:

1. Píxeles dentro de la extensión original y con las cinco bandas válidas y finitas.
2. Píxeles con referencia binaria interpretable según la leyenda del conjunto.
3. Píxeles aceptados por la máscara `VALID` publicada para Sentinel-2 (valor 1), una vez comprobada su correspondencia con la escena descargada.
4. Píxeles con ambos denominadores positivos y con probabilidad U-Net finita entre cero y uno.

No se supondrá que una etiqueta de fondo sea NoData ni se inferirá la validez solo a partir del color RGB. Los valores de agua/fondo y la máscara auxiliar se verificarán por separado. Si el producto o una distribución curada carecen de la validez necesaria, su uso requerirá una regla explícita previa y una declaración de esa limitación; no se inventará una comprobación de nubes.

Los tres métodos se medirán sobre ese mismo soporte. Se registrarán los píxeles excluidos y sus motivos, los recuentos de agua/fondo y las ventanas sin soporte evaluable. Si se obtienen provisionalmente métricas de los índices antes de disponer de U-Net, deberán identificarse como provisionales: el soporte pareado final depende también de la probabilidad válida de la red.

## Conteos, agrupación y resultados

Se guardarán VP, FP, FN y VN, IoU, F1, precisión y exhaustividad por ventana y por escena original, además del global. El global sumará conteos de ventanas no repetidas antes de dividir. No se obtendrá promediando porcentajes de los parches ni volviendo a sumar tablas agregadas.

Se conservará una bandera para métricas con denominador nulo. Por continuidad con el piloto, el valor computacional podrá ser cero junto a esa bandera; los casos sin agua deberán informar además los falsos positivos y el soporte. Una métrica no definida no debe interpretarse como una omisión total demostrada.

La unidad de agrupación será la escena/adquisición original, con una agrupación adicional de fechas o escenas relacionadas cuando proceda. Los parches de una misma escena no son réplicas independientes. La diferencia U-Net menos cada índice se expresará en puntos porcentuales, calculados a partir de valores sin redondear.

La descripción de los resultados es necesaria; los intervalos son opcionales. Si se calculan, se remuestrearán grupos completos y se usarán los mismos sorteos para todos los métodos. El toolkit previo exige al menos diez unidades de remuestreo con soporte y no certifica independencia por alcanzar ese número. No se sustituirá ese límite dividiendo una escena en más parches.

La conclusión evaluará desempeño externo bajo las condiciones realmente usadas. Una ventaja en este conjunto no demuestra que VGG16, B8 o la arquitectura concreta sean la causa, porque no se compararon esos componentes de forma aislada. Tampoco resuelve por sí sola el origen desconocido de las etiquetas históricas de El Oro ni la exposición previa de los cuatro recortes del Test original.

## Cambio necesario respecto de las herramientas de anotación propias

El evaluador existente en la ampliación fue diseñado para anotaciones propias y exige `reference_origin=human_blind`, congelación anterior a predicciones y óptica descrita como reflectancia de superficie. La procedencia de referencia y la declaración de cegamiento no corresponden automáticamente a S1S2-Water, aunque en esta ruta se recupere óptica L2A.

La implementación externa utiliza la procedencia explícita `public_expert_refined_ndwi`, con conjunto, versión, fuente, hash, leyenda y constancia de su inicialización NDWI. Registra la contraparte L2A, su correspondencia con la adquisición original y la selección congelada para esta evaluación. No se rellenará `human_blind` ni se afirmará una cronología de anotación propia solo para superar la validación del programa. Las funciones de métricas y soporte se adaptan a estos contratos de forma explícita y se prueban con fixtures sintéticos.

## Entrega que debe conservarse

| Archivo o resultado | Contenido mínimo |
|---|---|
| Selección congelada | Criterio, versión, identificadores, grupos, exclusiones y momento real de congelación. |
| Manifiesto de datos | URLs y hashes de referencia y validez; metadatos originales y correspondencia exacta de la contraparte L2A, sus bandas, radiometría y cuadrícula. |
| Registro del modelo | Archivo, SHA-256, arquitectura/capas personalizadas, entrada esperada y entorno de inferencia. |
| Probabilidades | Salidas U-Net alineadas y conservadas, sin transformarlas en referencia. |
| Tablas | Conteos y métricas por ventana, escena/grupo y global sobre soporte común. |
| Metadatos de ejecución | Fecha real, versiones, parámetros fijos, exclusiones y hashes de código/insumos. |
| Figuras | Comparación global y por escena; ejemplos seleccionados por regla previa que incluyan acuerdos y errores. |

## Prioridades

Son necesarias para esta ampliación la adquisición trazable, la correspondencia de adquisición y cuadrícula L2A, la comprobación del checkpoint, la selección previa, la compatibilidad radiométrica, la inferencia y las métricas comunes de los tres métodos. Las referencias publicadas con revisión experta permiten completar esta ruta sin reclutar anotadores propios, manteniendo explícito que fueron inicializadas con NDWI.

Quedan como ampliaciones opcionales el ajuste de umbrales en una validación separada, varias semillas, la comparación de codificadores, la ablación de B8 y una evaluación adicional con el L1C original. Cada una exige su propio diseño y no debe ajustar decisiones con el Test externo ya observado.

Las ocho ventanas locales preparadas para anotación quedan separadas de esta ruta. Sus plantillas no se completan artificialmente ni se presentan como anotaciones humanas. El manuscrito solo incorporará números de la nueva evaluación cuando los archivos de resultados y su procedencia estén disponibles.
