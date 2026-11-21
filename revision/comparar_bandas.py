"""Descriptive Train/external band comparison; never estimates Train from Test."""
import argparse,csv,json
from pathlib import Path

def run(train,external,out):
    train=json.loads(Path(train).read_text(encoding='utf-8-sig'))
    external=json.loads(Path(external).read_text(encoding='utf-8-sig'))
    if train.get('status')!='all_frozen_train_inputs_measured' or train.get('expected_cases')!=460:
        raise ValueError('Se requieren momentos medidos sobre los 460 Train reales.')
    if external['source_manifest_sha256']!='53c01804256a98a65d0ee212650f28153350d677da913fcc85d6d117beea8f97':
        raise ValueError('Estadísticos externos distintos del manifiesto congelado.')
    out=Path(out)
    if out.exists():raise ValueError('La salida ya existe; usar otro nombre.')
    source={r['band']:r for r in train['rows']};rows=[]
    for e in external['rows']:
        t=source[e['band']]
        if t['pixels']!=460*256*256:raise ValueError('Soporte Train incompleto.')
        rows.append({'band':e['band'],'external_support':e['scope'],'train_pixels':t['pixels'],'external_pixels':e['pixels'],
                     'train_mean':t['mean'],'external_mean':e['mean'],'external_minus_train_mean':e['mean']-t['mean'],
                     'train_std_population':t['std_population'],'external_std_population':e['std_population'],
                     'std_ratio_external_over_train':e['std_population']/t['std_population'] if t['std_population'] else None,
                     'units':'physical reflectance, descriptive ddof=0; no significance or causal attribution'})
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'Comparación descriptiva guardada: {out}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--train',required=True);p.add_argument('--external',default='analisis_local/estadisticos_bandas_externa.json');p.add_argument('--out',required=True);a=p.parse_args();run(a.train,a.external,a.out)
