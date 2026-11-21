"""Concilia los conteos, documenta los cambios de clase y prepara lectura por escena."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
from datetime import datetime, timezone

def read(base, name):
    with (base / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def pct(x): return f'{x * 100:.2f}'.replace('.', ',') + ' %'
def integer(x): return f'{int(x):,}'.replace(',', ' ')
def number(x): return f'{float(x):.4f}'.replace('.', ',')
def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--tablas", type=Path, required=True)
    args.add_argument("--externo", type=Path, required=True)
    args.add_argument("--salida", type=Path, required=True)
    options = args.parse_args()
    OUT, EXT = options.tablas, options.externo
    options.salida.mkdir(parents=True, exist_ok=True)
    scores = read(OUT, 'metricas_indices_por_escena.csv')
    controls = read(OUT, 'controles.csv')
    cases = read(OUT, 'metricas_indices_por_caso.csv')
    flips = read(OUT, 'cambios_clasificacion_por_escena.csv')
    bands = read(OUT, 'distribuciones_bandas_indices.csv')
    meta = json.loads((OUT/'diagnostico_metadatos.json').read_text(encoding='utf-8'))
    ext = {(r['aggregate_id'],r['method']):r for r in read(EXT, 'metricas_por_escena.csv')}
    assert len(scores)==12 and len(cases)==192 and len(controls)==48 and len(bands)==90 and len(flips)==18
    score = {(r['scene_id'], r['level'], r['index']): r for r in scores}
    flip = {(r['scene_id'],r['support'], r['index']): r for r in flips}
    bs = {(r['scene_id'],r['level'],r['variable'],r['support']):r for r in bands}
    receipt = {r['scene_id']:r for r in meta['source_files']}
    comparable = []
    for r in scores:
        n = int(r['valid_pixels'])
        assert sum(int(r[k]) for k in ['tp','fp','fn','tn']) == n
        group = [x for x in cases if (x['scene_id'],x['level'],x['index']) == (r['scene_id'],r['level'],r['index'])]
        assert len(group)==16
        for k in ['tp','fp','fn','tn','valid_pixels']:
            assert int(r[k]) == sum(int(v[k]) for v in group)
        if r['level']=='l2a':
            e = ext[r['scene_id'],r['index']]
            comparable.append({'escena':r['scene_id'], 'indice':r['index'], 'pixeles_soporte_diagnostico':n,
                               'pixeles_soporte_externo':int(e['valid_pixels']),
                               'conteos_identicos_ante_soporte_igual': (all(int(r[k]) == int(e[k]) for k in ['tp','fn','fp','tn'])
                                                                       if int(e['valid_pixels'])==n else None)})
            assert n <= int(e['valid_pixels'])
            assert comparable[-1]['conteos_identicos_ante_soporte_igual'] is not False
    for r in flips:
        assert int(r['pixels'])==sum(int(r[k]) for k in ['positive_both','l1c_positive_l2a_negative','l1c_negative_l2a_positive','negative_both'])
        sid,idx = r['scene_id'],r['index']
        if r['support'] != 'all':
            col='tp' if r['support']=='reference_water' else 'fp'
            assert int(r['positive_both'])+int(r['l1c_positive_l2a_negative'])==int(score[sid,'l1c',idx][col])
            assert int(r['positive_both'])+int(r['l1c_negative_l2a_positive'])==int(score[sid,'l2a',idx][col])
    header = """# Diagnóstico de las tres escenas con los productos ópticos originales

Las 48 ventanas y las referencias permanecen fijas. Las tres escenas se eligieron para investigar discrepancias ya observadas en la evaluación externa. Los resultados describen estos casos y no estiman el comportamiento del conjunto mundial completo.

Cada fila acumula los conteos de 16 ventanas. Dentro de cada escena, las cuatro variantes comparten exactamente los píxeles válidos. El umbral aplicado es estrictamente mayor que cero para ambos índices.

| Escena | Píxeles comunes | F1 NDWI L1C | F1 NDWI L2A | F1 MNDWI L1C | F1 MNDWI L2A |
|---|---:|---:|---:|---:|---:|
"""
    lines=[header.strip()]
    for sid in ['23','25','33']:
        row = [f'S{int(sid):03}',integer(score[sid,'l1c','ndwi']['valid_pixels'])]
        row += [pct(float(score[sid,level,idx]['f1'])) for idx in ['ndwi','mndwi'] for level in ['l1c','l2a']]
        lines.append('| '+' | '.join(row)+' |')
    for sid in ['23','25','33']:
        r=receipt[sid]
        sid_label=f'S{int(sid):03}'
        lines.extend(['',f'## Escena {sid_label}','',
                      f"Adquisición {r['same_acquisition_key'][1]}, tesela {r['same_acquisition_key'][3]}, órbita {r['same_acquisition_key'][2]}. Los identificadores completos y las huellas de los archivos están en diagnostico_metadatos.json.",''])
        for idx in ['ndwi','mndwi']:
            w=flip[sid,'reference_water',idx]; b=flip[sid,'reference_background',idx]
            n=int(w['pixels']); neg=int(w['negative_both'])
            med1=bs[sid,'l1c',idx,'reference_water']['median'];med2=bs[sid,'l2a',idx,'reference_water']['median']
            lines.append(f"{idx.upper()} tuvo una mediana de {number(med1)} en L1C y {number(med2)} en L2A entre los {integer(n)} píxeles etiquetados como agua. De L1C a L2A, {integer(w['l1c_positive_l2a_negative'])} pasaron de agua a fondo y {integer(w['l1c_negative_l2a_positive'])} de fondo a agua. Ambos niveles clasificaron {integer(neg)} como fondo ({pct(neg/n)} del agua de referencia). En el fondo de referencia, los falsos positivos pasaron de {integer(score[sid,'l1c',idx]['fp'])} a {integer(score[sid,'l2a',idx]['fp'])}.")
            lines.append('')
    lines += ["## Alcance de la comparación","","L1C proporciona reflectancia en el techo de la atmósfera. L2A proporciona reflectancia de superficie de la misma adquisición. Los TIFF de los autores se verificaron contra el tamaño y CRC de sus miembros ZIP y se identificaron con un SHA-256 local. La verificación abarca los tres miembros descargados. Las diferencias observadas incluyen el nivel de procesamiento y las decisiones de alineación y remuestreo de los productos.","","Wieland et al. describen la inicialización de las máscaras mediante NDWI y correcciones iterativas por tres expertos (sección III.C, p. 1089). El artículo no especifica el valor numérico del umbral inicial. El contraste a cero no permite cuantificar las ediciones históricas ni aislar una causa de los desacuerdos. [Artículo](https://doi.org/10.1109/JSTARS.2023.3333969) · [Conjunto v1.0.1](https://zenodo.org/records/11278238)","",
    "Las distribuciones de B3, B8, B11, NDWI y MNDWI se entregan por nivel, escena y clase de referencia, junto con extremos, media, desviación poblacional y cuantiles. Los controles y las transiciones por recorte permiten reproducir las agregaciones de esta lectura."]
    (options.salida/'lectura_escenas.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (options.salida/'conciliacion_soportes.json').write_text(json.dumps({'comprobado_utc':datetime.now(timezone.utc).isoformat(),
        'filas_matrices_por_recorte':len(cases),'filas_matrices_por_escena':len(scores),
        'transiciones_por_escena_conciliadas':len(flips),'distribuciones':len(bands),
        'comparacion_con_soporte_externo_previo':comparable},indent=2),encoding='utf-8')

if __name__=='__main__': main()
