"""Genera figuras únicamente después de una evaluación externa terminada."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = [('unet', 'U-Net / VGG16', '#235C7A'), ('ndwi', 'NDWI', '#AA752B'), ('mndwi', 'MNDWI', '#4A8060')]


def table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def defined(value):
    return str(value).lower() == 'true'


def generate(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'evaluacion_metadatos.json').read_text(encoding='utf-8'))
    if not (metadata.get('evaluation_executed') is True and metadata.get('inference_executed') is True
            and metadata.get('status') == 'external_evaluation_completed'):
        raise ValueError('Las figuras requieren una evaluación real completada, con predicciones del modelo.')
    global_rows = {r['method']: r for r in table(directory / 'metricas_globales.csv')}
    scene_rows = table(directory / 'metricas_por_escena.csv')
    if set(global_rows) != {m[0] for m in METHODS}:
        raise ValueError('Faltan métodos en la tabla global.')
    for key, _, _ in METHODS:
        if int(global_rows[key]['valid_pixels']) != int(global_rows['unet']['valid_pixels']):
            raise ValueError('Los métodos no tienen el mismo soporte global.')
    target = directory / 'figuras'
    target.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), layout='constrained')
    for ax, metric, title in zip(axes, ['iou', 'f1'], ['(a) IoU de agua', '(b) F1 de agua']):
        for x, (method, label, color) in enumerate(METHODS):
            row = global_rows[method]
            value = float(row[metric]) * 100
            if defined(row[metric + '_defined']):
                ax.bar(x, value, color=color, width=.58, zorder=3)
                ax.text(x, value + 2, f'{value:.2f}'.replace('.', ','), ha='center', va='bottom', fontsize=8)
            else:
                ax.text(x, 3, 'No definida', rotation=90, ha='center', va='bottom', fontsize=7)
        ax.set_xticks(range(3), [m[1] for m in METHODS], fontsize=8)
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel('Concordancia (%)')
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', color='#DDE1E3', linewidth=.6, zorder=0)
        ax.set_axisbelow(True)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(target / ('Figura_externa_1_comparacion_global.' + extension), dpi=600)
    plt.close(fig)

    ids = sorted({r['aggregate_id'] for r in scene_rows}, key=lambda x: int(x))
    indexed = {(r['aggregate_id'], r['method']): r for r in scene_rows}
    fig, ax = plt.subplots(figsize=(7.0, max(4.0, .27 * len(ids) + 1.0)), layout='constrained')
    y = np.arange(len(ids))
    for offset, (method, label, color) in zip([-.18, 0, .18], METHODS):
        values = []
        for scene in ids:
            row = indexed[scene, method]
            values.append(float(row['f1']) * 100 if defined(row['f1_defined']) else np.nan)
        ax.scatter(values, y + offset, s=23, color=color, label=label, zorder=3)
    labels = [f"S{int(scene):03d} (n = {indexed[scene, 'unet']['n_cases']})" for scene in ids]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-2, 102)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('F1 de agua por escena (%)')
    ax.grid(axis='x', color='#DDE1E3', linewidth=.6, zorder=0)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.1), ncol=3, frameon=False)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(target / ('Figura_externa_2_comparacion_escenas.' + extension), dpi=600)
    plt.close(fig)
    (target / 'LEEME.txt').write_text(
        'Fuente: ejecución externa conservada en el directorio padre.\n'
        'Figura 1: conteos globales acumulados antes de calcular IoU y F1.\n'
        'Figura 2: F1 calculado desde los conteos de cada escena; n indica recortes.\n'
        'Las métricas no definidas no se dibujan como valores observados.\n'
        'Referencia: S1S2-Water inicializado con NDWI y refinado por expertos; entrada L2A adaptada.\n'
        'Las figuras no incluyen intervalos de confianza ni prueban significación estadística.\n', encoding='utf-8')
    print(f'Figuras guardadas en {target}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True)
    generate(parser.parse_args().results)
