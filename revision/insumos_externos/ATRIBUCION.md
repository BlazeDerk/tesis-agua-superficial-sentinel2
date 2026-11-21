# Fuentes y atribución del subconjunto

Referencias de agua y validez: Wieland, M., Fichtner, F., Martinis, S., Groth, S., Krullikowski, C., Plank, S. y Motagh, M. **S1S2-Water**, versión 1.0.1, Zenodo, 24 de mayo de 2024. DOI del conjunto: [10.5281/zenodo.11278238](https://doi.org/10.5281/zenodo.11278238). La API del depósito identifica la licencia [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Se conserva su captura en `fuentes/zenodo_record_11278238.json`.

Artículo del conjunto: Wieland et al. (2024), *S1S2-Water: A Global Dataset for Semantic Segmentation of Water Bodies From Sentinel-1 and Sentinel-2 Satellite Images*. DOI: [10.1109/JSTARS.2023.3333969](https://doi.org/10.1109/JSTARS.2023.3333969).

Modificaciones realizadas para esta evaluación: extracción de 16 ventanas completas de 256 × 256 por cada escena Test mediante selección espacial SHA-256, sin alterar las clases de agua y validez. Se conserva el registro de extracción y hashes. Las referencias fueron inicializadas con NDWI y refinadas por expertos según el artículo; no se atribuyen a los autores del piloto ecuatoriano.

Óptica: datos Sentinel-2 Copernicus, nivel L2A, obtenidos de Sentinel COGs mediante el catálogo público [Earth Search](https://earth-search.aws.element84.com/v1). Las adquisiciones coinciden con las fuentes S1S2-Water por plataforma, instante de toma, órbita relativa y tesela MGRS. URLs y productos individuales figuran en `fuentes/l2a/` y en los registros de cada recorte. Se aplicaron escala y offset del activo una sola vez, alineación de las bandas de 10 m por vecino próximo e interpolación bilineal de B11 de 20 a 10 m. La licencia del conjunto de etiquetas no debe confundirse con las condiciones de los productos Sentinel-2 del proveedor.

El subconjunto adaptado no reemplaza al depósito original ni constituye una reproducción exacta de su benchmark L1C. Los scripts de este paquete fueron preparados con asistencia de Codex y requieren revisión de los autores antes de su publicación. Los pesos de U-Net no se redistribuyen en este paquete.
