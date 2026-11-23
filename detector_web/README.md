# AQUA · Detector de agua superficial

AQUA lee un GeoTIFF de Sentinel-2 y produce una máscara georreferenciada de agua superficial. La inferencia se ejecuta en el navegador con los pesos del checkpoint final. Esta edición sirve para repetir el análisis con un archivo propio. Las imágenes de ejemplo de la instalación del equipo y las vistas previas del legado de El Oro se conservan por separado.

## Qué se puede comparar

| Capa | Origen | Alcance |
| --- | --- | --- |
| Agua y fondo | U-Net/VGG16 con cuatro bandas y umbral de probabilidad ≥ 0,5 | El fondo reúne superficies como suelo, vegetación y construcciones. |
| Nubes y sombras | Clasificación auxiliar SCL de la misma adquisición | Las clases 8, 9 y 10 identifican nubes. La clase 3 identifica sombras. Sin SCL, la información de nubosidad queda sin evaluar. |
| Agua en posibles estanques | Agua predicha dentro de la unión de rectángulos candidatos | La geometría sirve para seleccionar lugares para revisión. Acreditar el uso camaronero exige una referencia externa. |

La interfaz permite cargar referencias por capa y comparar ambas delimitaciones sobre los píxeles válidos. Presenta IoU, F1, precisión y sensibilidad cuando sus denominadores están definidos. Los resultados incluyen mapas de discrepancia, máscaras GeoTIFF, probabilidades, candidatos GeoJSON, tabla e informe descargables. El origen de la referencia queda registrado en el informe.

## Ejecutar en el equipo

Se necesita Node.js 22 o posterior, Python 3.12 con las dependencias de inferencia y el archivo original `model_best.keras`. Los comandos se ejecutan desde `detector_web/`.

```sh
npm ci --ignore-scripts
npm run preparar
python -m pip install -r ../revision/requirements-revision.txt
python preparar_pesos.py "/ruta/al/model_best.keras"
python -m http.server 8000 --bind 127.0.0.1 --directory dist
```

Abre [el detector local](http://127.0.0.1:8000/). La dirección de bucle local habilita la verificación criptográfica de los pesos en el navegador. Para servirlo desde otro origen se requiere HTTPS. El equipo custodio de la tesis conserva el checkpoint. El exportador comprueba su SHA-256 y reconstruye las 48 matrices en ocho fragmentos que deben coincidir con el manifiesto. El conjunto de entrenamiento no se necesita para esta operación.

El instalador de JavaScript fija TensorFlow.js 4.22.0, GeoTIFF.js 3.0.5 y proj4 2.22.0. El preparador reproduce las dos sustituciones de compatibilidad con la política CSP de la edición publicada y comprueba las tres bibliotecas por SHA-256. Tras esta preparación, la interfaz carga todos sus componentes del mismo servidor. El material de terceros conserva sus atribuciones en [vendor](dist/vendor/) y [la licencia de la tipografía Manrope](dist/assets/FONT-LICENSE.txt).

## Archivos de entrada

- Óptica Sentinel-2 en GeoTIFF uint16 de 10 m, en EPSG:32717 y con B2, B3, B4 y B8 en ese orden. La reflectancia se representa con factor 10 000. Una quinta banda en el paquete de tesis se omite de la entrada de la red. Cada archivo óptico admite hasta 128 MiB y cuatro millones de píxeles.
- SCL opcional de una banda alineada con la imagen. Debe conservar los códigos categóricos de la misma adquisición. Una capa con el número de observaciones despejadas tiene un significado distinto.
- Las referencias opcionales se aportan como GeoTIFF binarios de una banda alineados a la misma cuadrícula. El valor 1 marca presencia, 0 ausencia y 255 o 65535 identifica píxeles sin revisar. La interfaz comprueba la alineación y pide declarar la procedencia de las etiquetas.

## Pruebas y evidencia

| Comprobación | Evidencia | Lectura |
| --- | --- | --- |
| Paridad entre Keras y WebGL | [Resultado de paridad](pruebas/paridad_keras_webgl.json) | En un recorte de validación, la diferencia máxima de probabilidad fue 0,00002635 y los 65 536 píxeles conservaron la misma clase a umbral 0,5. |
| Cinco recortes de validación | [Conteos y métricas por recorte](pruebas/cinco_recortes_validacion.csv) | Se seleccionó un recorte por zona y un quinto entre los restantes. Se comparó con las máscaras disponibles. |
| Cinco escenas ecuatorianas de 2025 | [Fecha, identificador y superficies por escena](pruebas/escenas_ecuador_2025.csv) | Prueba de uso en Esmeraldas, Daule–Peripa, Quilotoa, Cajas y el río Upano. La selección fue dirigida. Al carecer de máscaras humanas se reportan áreas predichas y soporte analizable. |
| Comparación agua, fondo, nubes y candidatos | [Registro de controles por capa](pruebas/comparacion_capas_20261008.json) | Incluye controles de alineación y valores desconocidos, y una comprobación funcional en Esmeraldas con SCL. |

Las pruebas de software utilizan los mismos casos de la aplicación. El primer comando ejecuta 19 comprobaciones de métricas y clases. El segundo realiza 30 controles de entrada e integridad después de preparar los pesos y las bibliotecas. El tercero construye el distribuidor HTTP opcional en `dist/server/` y comprueba ocho casos con recursos simulados.

```sh
npm run prueba:capas
npm run prueba:integridad
npm run prueba:http
```

La comparación histórica de 59 y 55 recortes y la prueba externa S1S2-Water con 16 escenas se documentan en la [revisión del artículo](../revision/README.md).

### Repetir la prueba en un navegador

El [registro de las cinco escenas](pruebas/procedencia_escenas_2025.json) conserva los identificadores STAC, enlaces a bandas y SCL, cuadrículas, conversión radiométrica y hashes de los TIFF. Para repetir la prueba completa de interfaz con una copia de los recortes óptico y SCL archivados de Esmeraldas, indica el ejecutable de Chrome o Edge y ambos archivos. El verificador comprueba sus hashes, carga las imágenes como lo haría un usuario, descarga el informe JSON y coteja los cuatro conteos con los de octubre.

```sh
npm install --prefix .sites-runtime/verificacion --ignore-scripts --no-audit playwright-core@1.56.1
node tests/escena.mjs "/ruta/al/navegador" "/ruta/esmeraldas_optical.tif" "/ruta/esmeraldas_scl.tif"
```

El ensayo del 23 de noviembre de 2026 reprodujo el informe de Esmeraldas con 6 235 píxeles de agua, 51 219 de fondo y 748 de nubes SCL. De los 6 235 píxeles de agua, 125 quedaron en candidatos a estanques. Todos los cálculos se ejecutaron en el equipo de prueba. [Verificación del navegador](pruebas/navegador_esmeraldas_20261123.json) · [Verificación de pesos y 57 controles](pruebas/controles_repositorio_20261123.json).
