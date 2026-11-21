"""Análisis posterior exploratorio: escena completa como unidad pareada.

No reentrena, modifica umbrales ni remuestrea píxeles. Lee los conteos de
las 16 escenas que ya fueron reproducidos desde los GeoTIFF conservados.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'insumos_externos/conteos'
OUT = ROOT / 'estadistica_recalculada'
METHODS = ['unet', 'ndwi', 'mndwi']
METRICS = ['iou', 'f1', 'precision', 'recall']
COUNTS = ['tp', 'fp', 'fn', 'tn']
REPS = 20000
SEED = 20261120


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(counts):
    counts = np.asarray(counts, dtype=np.float64)
    tp, fp, fn, tn = np.moveaxis(counts, -1, 0)
    numerator = np.stack([tp, 2*tp, tp, tp], axis=-1)
    denominator = np.stack([tp+fp+fn, 2*tp+fp+fn, tp+fp, tp+fn], axis=-1)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def paired_bootstrap(counts, reps=REPS, seed=SEED):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(counts), size=(reps, len(counts)), endpoint=False)
    # The same scene draw is applied simultaneously to all three methods.
    selected = counts[draws]
    micro = metrics(selected.sum(axis=1))
    macro = metrics(counts)[draws].mean(axis=1)
    return {'micro': micro, 'macro': macro}, draws


def exact_permutations(counts):
    n = len(counts)
    assignments = ((np.arange(2**n, dtype=np.uint32)[:, None] >> np.arange(n)) & 1).astype(np.int64)
    rows = []
    scene_metrics = metrics(counts)
    for j, baseline in [(1, 'ndwi'), (2, 'mndwi')]:
        a, b = counts[:, 0], counts[:, j]
        adjustment = assignments @ (b-a)
        swap_a, swap_b = a.sum(axis=0)+adjustment, b.sum(axis=0)-adjustment
        perm_micro = metrics(swap_a)-metrics(swap_b)
        micro_observed = metrics(a.sum(axis=0))-metrics(b.sum(axis=0))
        per_scene_difference = scene_metrics[:, 0]-scene_metrics[:, j]
        perm_macro = ((1-2*assignments) @ per_scene_difference)/n
        macro_observed = per_scene_difference.mean(axis=0)
        for estimand, observed, distribution in [('micro', micro_observed, perm_micro), ('macro', macro_observed, perm_macro)]:
            for k, name in enumerate(METRICS):
                p = np.count_nonzero(np.abs(distribution[:, k]) >= abs(observed[k])-1e-14)/len(assignments)
                rows.append({'estimand': estimand, 'comparison': 'unet_minus_'+baseline, 'metric': name,
                             'observed_difference_pp': float(observed[k]*100), 'exact_two_sided_p': float(p),
                             'n_scene_swaps_enumerated': int(len(assignments)), 'multiplicity_adjusted': False,
                             'status': 'exploratory_posthoc_conditional_on_scene_exchangeability'})
    return rows


def main():
    rows = read_csv(SOURCE/'metricas_por_escena.csv')
    ids = sorted({r['aggregate_id'] for r in rows}, key=int)
    assert len(ids) == 16 and len(rows) == 48
    lookup = {(r['aggregate_id'], r['method']): r for r in rows}
    counts = np.asarray([[[int(lookup[s,m][k]) for k in COUNTS] for m in METHODS] for s in ids], dtype=np.int64)
    assert np.all(counts.sum(axis=-1) == counts.sum(axis=-1)[:, :1])
    assert np.all((counts[:,:,0]+counts[:,:,2]) == (counts[:,:,0]+counts[:,:,2])[:, :1])
    scene_scores = metrics(counts)
    assert np.isfinite(scene_scores).all(), 'No convertir métricas de escena indefinidas a cero en una media.'
    global_rows = {r['method']:r for r in read_csv(SOURCE/'metricas_globales.csv')}
    for j,m in enumerate(METHODS):
        for k,c in enumerate(COUNTS):assert int(global_rows[m][c]) == counts[:,j,k].sum()
        for k,name in enumerate(METRICS):assert abs(metrics(counts.sum(axis=0))[j,k]-float(global_rows[m][name])) < 1e-14
    by_case = read_csv(SOURCE/'metricas_por_caso.csv')
    cluster_sets = {s:{r['cluster_id'] for r in by_case if r['scene_id']==s} for s in ids}
    assert all(len(x)==1 for x in cluster_sets.values()) and len({next(iter(x)) for x in cluster_sets.values()})==16
    distributions, draws = paired_bootstrap(counts)
    points = {'micro':metrics(counts.sum(axis=0)), 'macro':scene_scores.mean(axis=0)}
    summaries=[];comparisons=[]
    for estimand in ['micro','macro']:
        distribution=distributions[estimand]
        for j,m in enumerate(METHODS):
            for k,name in enumerate(METRICS):
                low,high=np.quantile(distribution[:,j,k],[.025,.975],method='linear')
                summaries.append({'estimand':estimand,'method':m,'metric':name,'point':float(points[estimand][j,k]),
                                  'ci95_low':float(low),'ci95_high':float(high),'bootstrap_reps':REPS,'scene_units':16,'seed':SEED,
                                  'status':'exploratory_scene_percentile_ci'})
        for j,baseline in [(1,'ndwi'),(2,'mndwi')]:
            diff=distribution[:,0]-distribution[:,j]
            for k,name in enumerate(METRICS):
                low,high=np.quantile(diff[:,k],[.025,.975],method='linear')
                point=points[estimand][0,k]-points[estimand][j,k]
                comparisons.append({'estimand':estimand,'comparison':'unet_minus_'+baseline,'metric':name,
                                    'difference_pp':float(point*100),'ci95_low_pp':float(low*100),'ci95_high_pp':float(high*100),
                                    'bootstrap_reps':REPS,'scene_units':16,'seed':SEED,
                                    'status':'exploratory_paired_scene_percentile_ci'})
    water=counts[:,0,0]+counts[:,0,2];total_water=int(water.sum());total_pixels=int(counts[:,0].sum())
    scenes=[]
    for i,s in enumerate(ids):
        row={'scene_id':s,'selected_cases':int(lookup[s,'unet']['n_cases']),
             'evaluable_cases':int(lookup[s,'unet']['n_cases_with_valid_pixels']),
             'valid_pixels':int(counts[i,0].sum()),'reference_water_pixels':int(water[i]),
             'reference_water_pct_of_global':float(100*water[i]/total_water),
             'reference_water_prevalence_pct':float(100*water[i]/counts[i,0].sum()),
             'all_three_f1_gt_96pct':bool(np.all(scene_scores[i,:,1]>.96))}
        for j,m in enumerate(METHODS):
            for k,name in enumerate(METRICS):row[m+'_'+name]=float(scene_scores[i,j,k])
            for k,c in enumerate(COUNTS):row[m+'_'+c]=int(counts[i,j,k])
        scenes.append(row)
    permutations=exact_permutations(counts)
    write_csv(OUT/'metricas_globales_y_medias_con_ic.csv',summaries)
    write_csv(OUT/'diferencias_pareadas_con_ic.csv',comparisons)
    write_csv(OUT/'escenas_y_pesos.csv',scenes)
    write_csv(OUT/'permutaciones_exploratorias.csv',permutations)
    weighting=[{'method':m,'metric':name,'global_pixels':float(points['micro'][j,k]),
                'unweighted_scene_mean':float(points['macro'][j,k]),
                'scene_mean_minus_global_pp':float(100*(points['macro'][j,k]-points['micro'][j,k]))}
               for j,m in enumerate(METHODS) for k,name in enumerate(METRICS)]
    write_csv(OUT/'cambio_por_ponderacion.csv',weighting)
    top_indices=np.argsort(water)[-4:][::-1]
    report={'analysis_type':'posthoc_exploratory', 'unit':'whole_scene_unique_cluster', 'scene_ids':ids,
            'methods':METHODS,'metrics':METRICS,'bootstrap_reps':REPS,'seed':SEED,
            'rng':'numpy.random.default_rng(PCG64)','numpy_version':np.__version__,
            'draw_indices_sha256':hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest(),
            'percentile_method':'linear 2.5th and 97.5th percentiles',
            'micro_definition':'sum integer scene confusion counts before calculating metric',
            'macro_definition':'unweighted mean of 16 defined scene metrics',
            'common_bootstrap_draws_for_all_methods':True,
            'global_valid_pixels':total_pixels,'global_reference_water_pixels':total_water,
            'largest_four_water_scenes':[ids[i] for i in top_indices],
            'largest_four_water_weight_pct':float(100*water[top_indices].sum()/total_water),
            'all_three_f1_gt_96pct':[r['scene_id'] for r in scenes if r['all_three_f1_gt_96pct']],
            'sources_sha256':{name:sha(SOURCE/name) for name in ['metricas_por_escena.csv','metricas_globales.csv','metricas_por_caso.csv']},
            'point_metrics':{k:{m:{name:float(points[k][j,h]) for h,name in enumerate(METRICS)} for j,m in enumerate(METHODS)} for k in points},
            'weighting_changes':weighting,
            'comparisons_with_ci':comparisons,'permutations':permutations,
            'limitations':['16 purpose-selected official Test scenes, not a random sample of the planet',
                           'scene bootstrap does not represent training seed variation or reference uncertainty',
                           'public labels initialized with NDWI and expert-refined; metric intervals do not remove anchoring',
                           'macro and micro answer different weighting questions; do not select estimand after seeing preferred winner',
                           'permutation p values conditional on within-scene exchangeability and not adjusted for multiple comparisons']}
    (OUT/'estadistica_escenas.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['ANÁLISIS EXPLORATORIO POR ESCENA','Unidad: 16 escenas, cada una un cluster distinto. Bootstrap pareado, 20 000 réplicas, semilla 20261120. IC95 percentil.','']
    for estimand in ['micro','macro']:
        lines.append(estimand.upper())
        for m in METHODS:
            selected=[r for r in summaries if r['estimand']==estimand and r['method']==m and r['metric'] in ['iou','f1']]
            lines.append(m+': '+ '; '.join(f"{r['metric']} {100*r['point']:.6f}% (IC95 {100*r['ci95_low']:.6f} a {100*r['ci95_high']:.6f})" for r in selected))
        for r in comparisons:
            if r['estimand']==estimand and r['metric'] in ['iou','f1']:
                lines.append(f"{r['comparison']} {r['metric']}: {r['difference_pp']:.6f} pp (IC95 {r['ci95_low_pp']:.6f} a {r['ci95_high_pp']:.6f})")
        lines.append('')
    for s in ['23','25','33']:
        row=next(r for r in scenes if r['scene_id']==s)
        lines.append(f"S{int(s):03d}: agua {row['reference_water_pixels']} px, peso {row['reference_water_pct_of_global']:.6f}%; F1 "+', '.join(f"{m}={100*row[m+'_f1']:.6f}%" for m in METHODS))
    lines.extend(['',f"Cuatro de mayor agua: {report['largest_four_water_scenes']}; peso={report['largest_four_water_weight_pct']:.6f}%.",
                  f"F1 >96% en los tres métodos: {report['all_three_f1_gt_96pct']}.",
                  'La media por escena favorece a U-Net; el global por píxeles favorece a los índices. Ninguno debe ocultar al otro.',
                  'Los intervalos son exploratorios, condicionados a estas escenas, referencias y al checkpoint fijo. No prueban independencia del índice, estabilidad entre semillas, exactitud de campo o generalización planetaria.'])
    (OUT/'brevelectura.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=False)
    main()
