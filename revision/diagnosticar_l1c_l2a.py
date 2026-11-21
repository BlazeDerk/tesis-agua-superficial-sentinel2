"""Diagnóstico pareado de L1C y L2A; no reconstruye la inicialización histórica.

Usa el TIFF S1S2-Water original de seis bandas, cuya integridad comprueba contra
el directorio central del ZIP. No descarga imágenes, carga modelos ni ajusta
umbrales. NDWI>0 y MNDWI>0 son reglas del diagnóstico actual, no umbrales
certificados de la inicialización de los autores.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,zlib
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

BANDS=('B2','B3','B4','B8','B11','B12')
EVAL_BANDS=BANDS[:5]

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def readjson(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def writejson(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def writecsv(path,rows):
    require(bool(rows),'No se escribe una tabla vacía.')
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def external_file(value,external_root,manifest_parent):
    value=Path(value)
    candidates=[value] if value.is_absolute() else [Path(external_root)/value,manifest_parent/value]
    if not value.is_absolute() and value.parts and value.parts[0]=='datos':candidates.append(Path(external_root).joinpath(*value.parts[1:]))
    matches={p.resolve() for p in candidates if p.is_file()}
    require(len(matches)==1,f'No se pudo resolver de forma única: {value}; indicar la raíz correcta de datos externos.')
    return matches.pop()

def original_member(root,scene):
    name=f'sentinel12_s2_{scene}_img.tif'
    matches={p.resolve() for p in Path(root).rglob(name) if p.is_file()}
    require(len(matches)==1,f'Se requiere una única copia original de {name}; encontradas {len(matches)}.')
    return matches.pop()

def archive_entry(index_root,scene):
    name=f'sentinel12_s2_{scene}_img.tif';matches=[]
    for path in Path(index_root).glob('part*.zip.central_directory.json'):
        data=readjson(path)
        for row in data['entries']:
            if Path(row['filename']).name==name:matches.append((path,data,row))
    require(len(matches)==1,f'No hay entrada ZIP única para {name}; conservar fuentes/part*.zip.central_directory.json del paquete.')
    return matches[0]

def verify_member(path,entry):
    require(path.stat().st_size==int(entry['file_size']),f'Tamaño distinto del miembro original: {path}')
    crc=0;digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):crc=zlib.crc32(block,crc);digest.update(block)
    crc&=0xffffffff
    require(crc==int(entry['CRC']),f'CRC32 distinto del miembro original: {path}')
    return {'bytes':path.stat().st_size,'crc32':crc,'sha256':digest.hexdigest()}

def aligned_l1c(source,reference):
    require(source.crs is not None and source.crs==reference.crs,'L1C y referencia deben declarar la misma proyección UTM del producto original.')
    require(source.count==6 and source.dtypes==('uint16',)*6,'Se requiere el IMG original de seis bandas UInt16.')
    require(source.res==(10,10) and reference.res==(10,10),'La cuadrícula publicada debe ser de10m; no se sustituye otra resolución.')
    require(source.transform.b==source.transform.d==reference.transform.b==reference.transform.d==0,'No se admite cuadrícula rotada sin inspección específica.')
    # Six-band order is fixed by the authors' specification. CRC checks bind
    # the file to their original member, even if descriptions are absent.
    return WarpedVRT(source,crs=reference.crs,transform=reference.transform,width=reference.width,height=reference.height,
                     dtype='float32',resampling=Resampling.nearest,nodata=65535.0)

def evaluate_pair(l1c,l2a,reference,validity,l1c_valid=None,l2a_valid=None):
    require(l1c.shape==l2a.shape and l1c.ndim==3 and l1c.shape[0]==5,'Se requieren dos arrays CHW de cinco bandas físicas.')
    shape=l1c.shape[1:];require(reference.shape==validity.shape==shape,'Dimensiones distintas.')
    require(np.isin(reference,[0,1]).all() and np.isin(validity,[0,1]).all(),'Referencia y validez deben ser0/1.')
    common=(validity==1)&np.isfinite(l1c).all(axis=0)&np.isfinite(l2a).all(axis=0)
    if l1c_valid is not None:common&=l1c_valid
    if l2a_valid is not None:common&=l2a_valid
    values={};predictions={}
    for level,a in [('l1c',l1c.astype('float64')),('l2a',l2a.astype('float64'))]:
        for index,other in [('ndwi',3),('mndwi',4)]:
            denominator=a[1]+a[other];common&=denominator>0
            values[level,index]=(a[1]-a[other],denominator)
    indices={}
    for key,(num,den) in values.items():
        indices[key]=np.divide(num,den,out=np.zeros(shape),where=common);predictions[key]=indices[key]>0
    metrics=[]
    for (level,index),prediction in predictions.items():
        truth=reference==1
        counts={'tp':int((common&truth&prediction).sum()),'fp':int((common&~truth&prediction).sum()),'fn':int((common&truth&~prediction).sum()),'tn':int((common&~truth&~prediction).sum())}
        row={'level':level,'index':index,'valid_pixels':int(common.sum()),'water_reference_pixels':counts['tp']+counts['fn'],**counts}
        for name,num,den in [('iou',counts['tp'],counts['tp']+counts['fp']+counts['fn']),('f1',2*counts['tp'],2*counts['tp']+counts['fp']+counts['fn']),('precision',counts['tp'],counts['tp']+counts['fp']),('recall',counts['tp'],counts['tp']+counts['fn'])]:row[name]=num/den if den else None
        metrics.append(row)
    flips=[]
    for index in ['ndwi','mndwi']:
        before,after=predictions['l1c',index],predictions['l2a',index]
        for label,mask in [('all',common),('reference_water',common&(reference==1)),('reference_background',common&(reference==0))]:
            flips.append({'index':index,'support':label,'pixels':int(mask.sum()),'positive_both':int((mask&before&after).sum()),'l1c_positive_l2a_negative':int((mask&before&~after).sum()),'l1c_negative_l2a_positive':int((mask&~before&after).sum()),'negative_both':int((mask&~before&~after).sum())})
    return common,indices,metrics,flips

def summarize_values(a):
    a=np.asarray(a,dtype='float64')
    if not a.size:return {'pixels':0,'mean':None,'std_population':None,'minimum':None,'q05':None,'q25':None,'median':None,'q75':None,'q95':None,'maximum':None}
    q=np.quantile(a,[.05,.25,.5,.75,.95])
    return {'pixels':int(a.size),'mean':float(a.mean()),'std_population':float(a.std(ddof=0)),'minimum':float(a.min()),'q05':float(q[0]),'q25':float(q[1]),'median':float(q[2]),'q75':float(q[3]),'q95':float(q[4]),'maximum':float(a.max())}

def run(args):
    manifest_path=Path(args.manifest).resolve();plan_path=Path(args.plan).resolve();plan=readjson(plan_path)
    require(sha(manifest_path)=='53c01804256a98a65d0ee212650f28153350d677da913fcc85d6d117beea8f97','El manifiesto no coincide con la selección externa congelada.')
    require(sha(plan_path)=='a3ebb2c24937c93f26626df87d32e87306daac6e1312d335989475a1f908f014','El plan no coincide con la selección previa a las predicciones.')
    require(plan['split']=='test','El plan debe ser el Test congelado.')
    with manifest_path.open(encoding='utf-8-sig',newline='') as f:manifest=list(csv.DictReader(f))
    scenes={str(x['scene_id']):x for x in plan['scenes']};cases={x['case_id']:x for x in plan['cases']}
    requested=[str(int(x)) for x in args.scene_ids.split(',')]
    require(len(set(requested))==len(requested) and all(s in scenes for s in requested),'IDs repetidos o ajenos al plan.')
    selected=[r for r in manifest if r['scene_id'] in requested]
    require(len(selected)==len({r['case_id'] for r in selected})==16*len(requested),'Se requieren las 16 ventanas únicas fijadas por cada escena; no una selección por desempeño.')
    for scene in requested:
        require({r['case_id'] for r in selected if r['scene_id']==scene}=={c['case_id'] for c in plan['cases'] if str(c['scene_id'])==scene},'Las ventanas no coinciden con todas las fijadas en el plan para la escena.')
    for row in selected:
        require(row['case_id'] in cases,'Caso fuera del plan.')
        for key in ['scene_id','row_off','col_off','height','width','cluster_id']:require(str(row[key])==str(cases[row['case_id']][key]),f'{row["case_id"]}: {key} distinto del plan.')
        require(row['l1c_source']==scenes[row['scene_id']]['l1c_source'] and row['l2a_product']==scenes[row['scene_id']]['l2a_product'],'Producto distinto del fijado.')
    out=Path(args.out_dir).resolve();require(not out.exists(),'La salida ya existe; usar un directorio nuevo.')
    index_root=Path(args.archive_index_root).resolve() if args.archive_index_root else plan_path.parent/'fuentes'
    receipts=[];metrics=[];flips=[];quantities=defaultdict(list);controls=[]
    out.mkdir(parents=True)
    for scene in requested:
        original=original_member(args.l1c_root,scene);entry_path,archive,entry=archive_entry(index_root,scene)
        fingerprint=verify_member(original,entry)
        ref_path=external_file(scenes[scene]['mask_path'],args.external_root,plan_path.parent)
        require(sha(ref_path)==scenes[scene]['mask_sha256'],'La referencia maestra no coincide con el plan.')
        item=external_file(scenes[scene]['item_path'],args.external_root,plan_path.parent)
        require(sha(item)==scenes[scene]['item_sha256'],'El registro L2A no coincide con el plan.')
        record={'scene_id':scene,'l1c_path':str(original),'l1c_source':scenes[scene]['l1c_source'],'l2a_product':scenes[scene]['l2a_product'],'l2a_id':scenes[scene]['l2a_id'],'same_acquisition_key':scenes[scene]['same_acquisition_key'],'l2a_item_sha256':sha(item),'original_member':entry['filename'],'archive_url':archive['url'],'archive_index_sha256':sha(entry_path),**fingerprint}
        with ExitStack() as stack:
            source=stack.enter_context(rasterio.open(original));master=stack.enter_context(rasterio.open(ref_path));view=stack.enter_context(aligned_l1c(source,master))
            record.update(native_crs=str(source.crs),native_shape=list(source.shape),native_transform=list(source.transform),native_band_descriptions=list(source.descriptions),original_order_from_author_spec=list(BANDS),source_nodata=source.nodata,reference_transform=list(master.transform),alignment='nearest_neighbor_to_unchanged_reference_grid',radiometry='uint16_DN/10000_once_no_offset_no_clipping')
            for row in [r for r in selected if r['scene_id']==scene]:
                paths={kind:external_file(row[kind+'_path'],args.external_root,manifest_path.parent) for kind in ['optical','reference','validity']}
                for kind,path in paths.items():require(sha(path)==row[kind+'_sha256'],f'Hash distinto: {path}')
                with rasterio.open(paths['optical']) as opt,rasterio.open(paths['reference']) as ref,rasterio.open(paths['validity']) as val:
                    require(opt.descriptions==EVAL_BANDS and opt.count==5 and opt.dtypes==('float32',)*5,'ÓpticaL2A: cinco bandas físicas float32 en ordenB2/B3/B4/B8/B11.')
                    require(opt.transform==ref.transform==val.transform and opt.crs==ref.crs==val.crs and opt.shape==ref.shape==val.shape==(256,256),'Recortes desalineados.')
                    require(opt.transform==master.window_transform(Window(int(row['col_off']),int(row['row_off']),256,256)),'Posición del recorte distinta de la referencia maestra.')
                    l2a=opt.read();refdata=ref.read(1);valid=val.read(1);l2valid=(opt.read_masks()!=0).all(axis=0)&(ref.read_masks(1)!=0)&(val.read_masks(1)!=0)
                win=Window(int(row['col_off']),int(row['row_off']),256,256)
                raw=view.read([1,2,3,4,5],window=win,masked=True).astype('float32');l1valid=~np.ma.getmaskarray(raw).any(axis=0);l1c=(raw/np.float32(10000.)).filled(np.nan)
                common,indices,scores,changes=evaluate_pair(l1c,l2a,refdata,valid,l1valid,l2valid)
                identifiers={'case_id':row['case_id'],'scene_id':scene}
                metrics.extend({**identifiers,**s} for s in scores);flips.extend({**identifiers,**s} for s in changes)
                controls.append({**identifiers,'total_pixels':65536,'diagnostic_common_pixels':int(common.sum()),'excluded_pixels':int(common.size-common.sum())})
                for support,mask in [('all',common),('reference_water',common&(refdata==1)),('reference_background',common&(refdata==0))]:
                    for level,a in [('l1c',l1c),('l2a',l2a)]:
                        for band,b in [('B3',1),('B8',3),('B11',4)]:quantities[scene,level,band,support].append(a[b,mask])
                        for index in ['ndwi','mndwi']:quantities[scene,level,index,support].append(indices[level,index][mask])
                print(f'Diagnóstico escena{scene}: {row["case_id"]}',flush=True)
        receipts.append(record)
    require(sum(r['diagnostic_common_pixels'] for r in controls)>0,'No hay soporte diagnóstico; no se presentan resultados vacíos.')
    summary=[]
    for (scene,level,variable,support),pieces in quantities.items():summary.append({'scene_id':scene,'level':level,'variable':variable,'support':support,**summarize_values(np.concatenate(pieces))})
    # Aggregate integer confusion counts before calculating each scene ratio.
    by_scene=[]
    for scene in requested:
        for level in ['l1c','l2a']:
            for index in ['ndwi','mndwi']:
                group=[r for r in metrics if r['scene_id']==scene and r['level']==level and r['index']==index];c={k:sum(r[k] for r in group) for k in ['tp','fp','fn','tn']}
                r={'scene_id':scene,'level':level,'index':index,'n_cases':len(group),'valid_pixels':sum(c.values()),**c}
                for name,num,den in [('iou',c['tp'],c['tp']+c['fp']+c['fn']),('f1',2*c['tp'],2*c['tp']+c['fp']+c['fn']),('precision',c['tp'],c['tp']+c['fp']),('recall',c['tp'],c['tp']+c['fn'])]:r[name]=num/den if den else None
                by_scene.append(r)
    flips_scene=[]
    for scene in requested:
        for index in ['ndwi','mndwi']:
            for support in ['all','reference_water','reference_background']:
                group=[r for r in flips if r['scene_id']==scene and r['index']==index and r['support']==support]
                flips_scene.append({'scene_id':scene,'index':index,'support':support,**{k:sum(r[k] for r in group) for k in ['pixels','positive_both','l1c_positive_l2a_negative','l1c_negative_l2a_positive','negative_both']}})
    writecsv(out/'metricas_indices_por_caso.csv',metrics);writecsv(out/'metricas_indices_por_escena.csv',by_scene);writecsv(out/'cambios_clasificacion.csv',flips);writecsv(out/'cambios_clasificacion_por_escena.csv',flips_scene);writecsv(out/'distribuciones_bandas_indices.csv',summary);writecsv(out/'controles.csv',controls)
    writejson(out/'diagnostico_metadatos.json',{'status':'paired_L1C_L2A_diagnostic_completed','created_utc':datetime.now(timezone.utc).isoformat(),'scene_ids':requested,'n_cases':len(selected),'manifest_sha256':sha(manifest_path),'plan_sha256':sha(plan_path),'thresholds':{'ndwi':'>0','mndwi':'>0'},'same_common_support_for_four_index_variants':True,'support_definition':'finite five-band values at both levels; manual VALID=1 and raster masks; positive denominators of both indices at both levels','reference_unmodified':True,'model_loaded':False,'threshold_tuning':False,'original_author_initial_threshold':'not specified in source available; zero is our diagnostic, not a certified reconstruction','interpretation_limit':'Level differences, historical expert edits, resampling or initialization parameters cannot be causally separated from this diagnostic alone. Do not claim experts rewrote the reference.','source_files':receipts})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene-ids',default='23,25,33');p.add_argument('--l1c-root',required=True);p.add_argument('--external-root',required=True);p.add_argument('--manifest',required=True);p.add_argument('--plan',required=True);p.add_argument('--archive-index-root');p.add_argument('--out-dir',required=True);run(p.parse_args())
