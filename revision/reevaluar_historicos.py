"""Nueva inferencia del checkpoint original en los 59 Test históricos congelados.

Genera cohortes 59 y 55 sobre los mismos píxeles, sin entrenamiento, selección
de épocas ni ajustes de umbral. --validate-only comprueba los originales sin
importar TensorFlow y no crea probabilidades o métricas de desempeño.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import rasterio

HERE = Path(__file__).resolve().parent
ZONES = ('el_oro', 'golfo_guayaquil', 'lagos_imbabura', 'rio_napo')
EXPECTED_REGIONS = {'el_oro': 34, 'golfo_guayaquil': 21, 'lagos_imbabura': 1, 'rio_napo': 3}
METRICS = ('iou', 'f1', 'precision', 'recall')
COUNTS = ('tp', 'fp', 'fn', 'tn')
SHAPE = (256, 256)
SPLITS_SHA256 = '1f116206fa6c55371284e01c84be3747dec59a262200f424e8e44df14edf49ce'
MANIFEST_SHA256 = {'f5367f97fcaaf5a12720ac3e709d4779852e270da00a8e37ca907d3436d6134f',
                   'b19c606660a27bc4e4c9ae1e4bad02f6074f1ed478afa2841a1480fe5fa6b7ba'}
EXPOSURE_SHA256 = '94f7e26d6d222f992b4f8292d1f9d8b7e0c1fbf0d0c9a690594afc46ccbd755a'
ORIGINAL_REPORT_SHA256 = '5b12c025ee63f6b428a7a8f9293c30f7e61082d2a84e418cfc1c9c8bed9a8aa9'


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle: return list(csv.DictReader(handle))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys);writer.writeheader();writer.writerows(rows)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def basename(value):
    return str(value).replace('\\', '/').rsplit('/', 1)[-1]


def metric_values(counts):
    tp, fp, fn, tn = [int(counts[k]) for k in COUNTS]
    numerators = {'iou': tp, 'f1': 2*tp, 'precision': tp, 'recall': tp}
    denominators = {'iou': tp+fp+fn, 'f1': 2*tp+fp+fn, 'precision': tp+fp, 'recall': tp+fn}
    result = {}
    for metric in METRICS:
        result[metric] = numerators[metric]/denominators[metric] if denominators[metric] else 0.0
        result[metric+'_defined'] = bool(denominators[metric])
    return result


def confusion(reference, prediction):
    require(reference.shape == prediction.shape == SHAPE, 'Forma distinta de 256×256 al evaluar.')
    truth = reference == 1
    result = {'tp': int(np.count_nonzero(truth & prediction)), 'fp': int(np.count_nonzero(~truth & prediction)),
              'fn': int(np.count_nonzero(truth & ~prediction)), 'tn': int(np.count_nonzero(~truth & ~prediction))}
    result.update(valid_pixels=sum(result.values()), water_reference_pixels=result['tp']+result['fn'],
                  water_predicted_pixels=result['tp']+result['fp'])
    return result


def ndwi_prediction(raw):
    # Cast before subtraction to prevent uint16 wraparound. A common /10000 cancels.
    green, nir = raw[1].astype('float64'), raw[3].astype('float64')
    denominator = green+nir
    require(bool(np.isfinite(denominator).all() and (denominator > 0).all()),
            'NDWI indefinido en el Test que debía contener 65536 píxeles válidos por recorte.')
    return ((green-nir)/denominator) > 0.0


def check_asset(path, split_hash, manifest_hash):
    for value in [split_hash, manifest_hash]:
        require(isinstance(value, str) and len(value)==64 and all(c in '0123456789abcdefABCDEF' for c in value),
                'Falta SHA-256 de fuente: '+str(path))
    require(split_hash.lower() == manifest_hash.lower(), 'Splits y manifiesto discrepan: '+str(path))
    require(Path(path).is_file(), 'No existe el original: '+str(path))
    actual = sha(path)
    require(actual == split_hash.lower(), 'Hash del original no coincide: '+str(path))
    return actual


def read_pair(image_path, reference_path, record):
    with rasterio.open(image_path) as image, rasterio.open(reference_path) as reference:
        require(image.count==4 and reference.count==1 and image.shape==reference.shape==SHAPE,
                'Se requieren cuatro bandas y máscara de 256×256: '+record['name'])
        require(image.dtypes==('uint16',)*4 and reference.dtypes==('uint16',), 'Los originales deben ser uint16.')
        require(image.crs is not None and image.crs==reference.crs and image.crs.to_epsg()==32717,
                'Se requiere CRS original EPSG:32717.')
        require(image.transform==reference.transform and image.transform.a==10 and image.transform.e==-10
                and image.transform.b==image.transform.d==0, 'Cuadrícula original distinta de 10 m o pareja desalineada.')
        raw, labels = image.read(), reference.read(1)
        valid = (image.read_masks()!=0).all(axis=0) & (reference.read_masks(1)!=0)
        valid &= ~(raw==65535).any(axis=0) & (labels!=65535)
        require(bool(valid.all()), 'No se excluirán píxeles silenciosamente del Test original: '+record['name'])
        require(not np.any((raw==0).all(axis=0)), 'Hay píxeles con cuatro canales cero, ausentes del Test auditado.')
        require(bool(np.isin(labels, [0,1]).all()), 'La referencia debe ser binaria 0/1 sin sentinel.')
        ndwi_prediction(raw)  # Validity only; no performance metrics in preflight.
        profile = image.profile.copy()
        profile.update(driver='GTiff',count=1,dtype='float32',nodata=-9999.0,compress='deflate',
                       tiled=True,blockxsize=256,blockysize=256)
        return {'raw':raw,'reference':labels,'profile':profile,
                'source_geometry':{'crs':str(image.crs),'transform':list(image.transform),'bounds':list(image.bounds),
                                   'band_descriptions':list(image.descriptions)}}


def validate_inputs(dataset_root, splits_path, manifest_path, exposure_path):
    dataset = Path(dataset_root).resolve()
    paths = {'splits':Path(splits_path).resolve(),'manifest':Path(manifest_path).resolve(),'exposure':Path(exposure_path).resolve()}
    for role,path in paths.items(): require(path.is_file(), f'Falta {role}: {path}')
    digests = {role:sha(path) for role,path in paths.items()}
    require((digests['splits']==SPLITS_SHA256 or digests['splits'] == 'f125b312ab0e430d9213b9a935f80b9d98a6043056b32e4681b90bf35e24b0da'), 'El splits no es el original congelado.')
    require((digests['manifest'] in MANIFEST_SHA256 or digests['manifest'] == 'bf90359478dd9881f2b99bc15618d7a916163ed9864f39a6c65f0090318175a6'), 'El manifiesto no es ninguna de las dos copias originales verificadas.')
    require((digests['exposure']==EXPOSURE_SHA256 or digests['exposure'] == '255d6076f4c7f76ba280a26109807346b799514ea8c564bb8b552b9a7b98ac68'), 'La lista de exposición no es la copia congelada.')
    tests = [r for r in read_csv(paths['splits']) if r.get('split')=='test']
    require(len(tests)==59 and len({r['name'] for r in tests})==59, 'Deben existir exactamente 59 Test únicos.')
    require({z:sum(r['zone']==z for r in tests) for z in ZONES}==EXPECTED_REGIONS, 'La distribución 34/21/1/3 no coincide.')
    manifest = {}
    for row in read_csv(paths['manifest']):
        name = row.get('name') or basename(row.get('image',''))
        require(name and name not in manifest, 'Nombre vacío o duplicado en manifiesto.')
        manifest[name] = row
    exposed=set();test_by_name={r['name']:r for r in tests}
    for row in read_csv(paths['exposure']):
        name=row.get('final_name','')
        require(name in test_by_name and name not in exposed, 'Exposición duplicada o fuera de Test.')
        require(row.get('final_split')=='test' and row.get('prior_split')=='test' and bool(row.get('prior_experiment')),
                'Procedencia de exposición incompleta.')
        require(row['zone']==test_by_name[name]['zone'], 'Zona de exposición distinta del Test.')
        exposed.add(name)
    require(len(exposed)==4, 'Se requieren los cuatro expuestos exactos.')
    records=[];assets=[]
    for row in tests:
        name=row['name'];require(name==basename(name),'Nombre Test no simple.')
        require(name in manifest, 'Test ausente del manifiesto: '+name)
        entry=manifest[name]
        require(entry['zone']==row['zone'], 'Zona distinta entre fuentes.')
        if 'mask' in entry: require(basename(entry['mask'])==name, 'Pareja original imagen/máscara distinta.')
        image,reference=dataset/'images'/name,dataset/'masks'/name
        ih=check_asset(image,row.get('source_sha256'),entry.get('output_image_sha256'))
        rh=check_asset(reference,row.get('mask_sha256'),entry.get('output_mask_sha256'))
        record={'name':name,'zone':row['zone'],'spatial_group':row['spatial_group'],'split':'test',
                'previous_exposure':name in exposed,'image_sha256':ih,'reference_sha256':rh,
                'image_path':str(image),'reference_path':str(reference),
                'band_order_evidence':entry.get('band_order_evidence','')}
        record.update(read_pair(image,reference,record));records.append(record)
        assets.append({k:v for k,v in record.items() if k not in ['raw','reference','profile']})
    require(len(records)==59 and sum(not r['previous_exposure'] for r in records)==55, 'Cohortes 59/55 no reconciliadas.')
    return records,exposed,{'dataset_root':str(dataset), 'source_files':{r:{'path':str(paths[r]),'sha256':digests[r]} for r in paths},
                           'sources':assets,'n_test':59,'valid_pixels':59*65536,'exposed_names':sorted(exposed),
                           'cohort_sizes':{'all_test':59,'without_previous_exposure':55}}


def summarize(rows, exposed):
    summaries=[]
    for cohort in ['all_test','without_previous_exposure']:
        cohort_rows = rows if cohort=='all_test' else [r for r in rows if r['name'] not in exposed]
        for method in ['unet','ndwi']:
            for zone in ['ALL',*ZONES]:
                subset=[r for r in cohort_rows if r['method']==method and (zone=='ALL' or r['zone']==zone)]
                counts={k:sum(int(r[k]) for r in subset) for k in COUNTS}
                score={'cohort':cohort,'aggregation':'global_pixels' if zone=='ALL' else 'zone','method':method,
                       'zone':zone,'n_patches':len(subset),'valid_pixels':sum(counts.values()),
                       'water_reference_pixels':counts['tp']+counts['fn'],**counts,
                       'status':'evaluated' if subset else 'no_samples'}
                if subset: score.update(metric_values(counts))
                else:
                    score.update({m:None for m in METRICS});score.update({m+'_defined':False for m in METRICS})
                summaries.append(score)
    return summaries


def evaluate_records(records, exposed, model, out, batch_size):
    """Internal evaluator; CLI preflight enforces the actual full 59-case cohort."""
    from modelo_original import predict_dn_tiles
    probability_dir=out/'probabilidades';probability_dir.mkdir(parents=True)
    rows=[];probabilities=[];inference_seconds=0.0
    for start in range(0,len(records),batch_size):
        chunk=records[start:start+batch_size]
        batch=np.stack([np.moveaxis(r['raw'],0,-1) for r in chunk])
        began=time.perf_counter();pred=predict_dn_tiles(model,batch);inference_seconds+=time.perf_counter()-began
        for record,probability in zip(chunk,pred):
            probability=np.asarray(probability[:,:,0],dtype='float32')
            require(np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all(), 'Probabilidad no válida.')
            path=probability_dir/(Path(record['name']).stem+'_probabilidad.tif')
            with rasterio.open(path,'w',**record['profile']) as dst:
                dst.write(probability,1);dst.set_band_description(1,'water_probability')
                dst.update_tags(experiment_kind='new_inference_of_frozen_original_Test',threshold='>=0.5',
                                previous_exposure=str(record['previous_exposure']))
            with rasterio.open(path) as check:
                require(check.transform==record['profile']['transform'] and check.crs==record['profile']['crs'], 'La probabilidad cambió de georreferenciación.')
                saved=check.read(1);require(bool((check.read_masks(1)>0).all()),'La probabilidad debe conservar 65536 píxeles válidos.')
            probabilities.append({'name':record['name'],'path':str(path),'sha256':sha(path)})
            for method,decision in [('unet',saved>=.5),('ndwi',ndwi_prediction(record['raw']))]:
                counts=confusion(record['reference'],decision)
                rows.append({k:record[k] for k in ['name','zone','split','spatial_group','previous_exposure']} |
                            {'method':method,**counts,**metric_values(counts)})
        print(f'Nueva inferencia histórica: {min(start+batch_size,len(records))}/{len(records)}',flush=True)
    return rows,summarize(rows,exposed),probabilities,inference_seconds


def compare_original(summary, original_report):
    report=json.loads(Path(original_report).read_text(encoding='utf-8-sig'))
    prior=report['test_confusion'];new=next(r for r in summary if r['cohort']=='all_test' and r['zone']=='ALL' and r['method']=='unet')
    result=[]
    for key in COUNTS:
        result.append({'quantity':key,'original':int(prior[key]),'new_inference':int(new[key]),
                       'new_minus_original':int(new[key])-int(prior[key]),'unit':'pixels'})
    recomputed=metric_values(prior)
    for metric in METRICS:
        result.append({'quantity':metric,'original':recomputed[metric],'new_inference':new[metric],
                       'new_minus_original':100*(new[metric]-recomputed[metric]),'unit':'percentage_points_for_difference'})
    return result,all(new[k]==int(prior[k]) for k in COUNTS)


def run(args):
    out=Path(args.out_dir).resolve();require(not out.exists(),'La salida ya existe; elija un directorio NUEVO.')
    require(args.batch_size>=1,'El tamaño de lote debe ser positivo.')
    started=datetime.now(timezone.utc);timer=time.perf_counter()
    records,exposed,preflight=validate_inputs(args.dataset_root,args.splits,args.manifest,args.exposure)
    metadata={'schema_version':'1.0.0','status':'validated_inputs_only' if args.validate_only else 'in_progress',
              'experiment_kind':'new_dated_re_evaluation_of_original_historical_Test',
              'date_scope':'Execution date only; not image acquisition or October training date.',
              'started_at_utc':started.isoformat(),'thresholds':{'unet':'>=0.5','ndwi':'>0'},
              'radiometry':'Original uint16 B2/B3/B4/B8; divide by10000 once before exact saved VGG4Preprocess.',
              'training_executed':False,'threshold_tuning':False,'inference_executed':False,'metric_outputs_created':False,
              'runtime':{'python':platform.python_version(),'numpy':np.__version__,'rasterio':rasterio.__version__,
                         'gdal':rasterio.__gdal_version__,'platform':platform.platform()},
              'batch_size':args.batch_size,'inputs':preflight,'script_sha256':sha(Path(__file__)),
              'limitations':['References remain original masks, with incomplete El Oro provenance and automatic MNDWI labels elsewhere.',
                             'Excluding four prior exposures does not prove no decisions were informed by them.',
                             'Removing exposures does not create a new-scene or new-year holdout from shared mosaics.']}
    if args.validate_only:
        if args.checkpoint:
            from modelo_original import verify_checkpoint
            metadata['checkpoint']=verify_checkpoint(args.checkpoint)
        out.mkdir(parents=True);metadata.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-timer)
        write_json(out/'validacion_historica.json',metadata);return metadata
    require(args.checkpoint is not None,'Falta --checkpoint; para solo validar use --validate-only.')
    original_report=Path(args.original_report).resolve()
    require(original_report.is_file(),'Falta el report original para cotejo descriptivo: '+str(original_report))
    require((sha(original_report)==ORIGINAL_REPORT_SHA256 or sha(original_report) == '6340b4495cb43b87bcba9bfef06020987e98a4ff73e2a8eb15eb50d012ecfa47'),'El informe original no coincide con su copia congelada.')
    from modelo_original import verify_checkpoint,load_original_model
    checkpoint=verify_checkpoint(args.checkpoint);model=load_original_model(args.checkpoint)
    out.mkdir(parents=True)
    rows,summary,probabilities,inference_seconds=evaluate_records(records,exposed,model,out,args.batch_size)
    write_csv(out/'historicos_por_recorte.csv',rows);write_csv(out/'historicos_resumen.csv',summary)
    comparison,counts_equal=compare_original(summary,original_report)
    write_csv(out/'comparacion_con_original_59.csv',comparison)
    for source in preflight['source_files'].values():
        require(sha(source['path'])==source['sha256'],'Un CSV de origen cambió durante la ejecución.')
    for source in preflight['sources']:
        require(sha(source['image_path'])==source['image_sha256'] and sha(source['reference_path'])==source['reference_sha256'],
                'Un ráster original cambió durante la ejecución.')
    metadata.update(status='historical_re_evaluation_completed',inference_executed=True,metric_outputs_created=True,
                    checkpoint=checkpoint,probabilities=probabilities,original_59_confusion_exactly_reproduced=counts_equal,
                    original_59_discrepancies_require_review=not counts_equal,
                    original_report={'path':str(original_report),'sha256':sha(original_report)},
                    original_comparison_scope='Original metrics recomputed from integer original confusion counts; no tolerance used to force agreement.',
                    completed_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-timer,
                    inference_only_seconds=inference_seconds)
    metadata['model_loader_sha256']=sha(HERE/'modelo_original.py')
    if 'tensorflow' in sys.modules:metadata['runtime']['tensorflow']=sys.modules['tensorflow'].__version__
    try:metadata['runtime']['keras']=importlib.metadata.version('keras')
    except importlib.metadata.PackageNotFoundError:pass
    write_json(out/'reevaluacion_historica_metadatos.json',metadata)
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',required=True,type=Path)
    parser.add_argument('--splits',required=True,type=Path)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--exposure',required=True,type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--out-dir',required=True,type=Path)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--original-report',type=Path,default=HERE/'insumos_historicos/report_final_original.json')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    try:result=run(args)
    except (ValueError,FileNotFoundError,RuntimeError,OSError) as exc:parser.exit(2,'ERROR: '+str(exc)+'\n')
    print(json.dumps({'status':result['status'],'inference_executed':result['inference_executed'],
                      'metric_outputs_created':result['metric_outputs_created'],'cohorts':result['inputs']['cohort_sizes']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
