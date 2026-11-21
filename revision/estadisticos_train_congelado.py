"""Compute exact frozen Train input moments; requires the real original rasters.

Example:
python estadisticos_train_congelado.py --splits ".../splits.csv" --images-dir ".../images" --out train_bandas.json
No training, augmentation, resampling, threshold tuning or model loading occurs.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import argparse,csv,hashlib,json
import numpy as np
import rasterio

BANDS=['B2','B3','B4','B8']

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class Moments:
    def __init__(self):self.n=0;self.mean=np.zeros(4);self.m2=np.zeros(4);self.minimum=np.full(4,np.inf);self.maximum=np.full(4,-np.inf)
    def add(self,a):
        a=np.asarray(a,dtype='float64').reshape(4,-1);n=a.shape[1];mu=a.mean(axis=1);m2=((a-mu[:,None])**2).sum(axis=1);delta=mu-self.mean;total=self.n+n
        self.m2+=m2+delta**2*self.n*n/total;self.mean+=delta*n/total;self.n=total
        self.minimum=np.minimum(self.minimum,a.min(axis=1));self.maximum=np.maximum(self.maximum,a.max(axis=1))
    def result(self):
        return [{'band':b,'pixels':self.n,'mean':float(self.mean[i]),'std_population':float(np.sqrt(self.m2[i]/self.n)),'minimum':float(self.minimum[i]),'maximum':float(self.maximum[i])} for i,b in enumerate(BANDS)]

def run(splits,images_dir,out,inventory_only=False):
    splits=Path(splits).resolve();out=Path(out).resolve()
    if out.exists():raise ValueError('La salida ya existe; conservar versiones y elegir otro nombre.')
    if sha(splits)!='1f116206fa6c55371284e01c84be3747dec59a262200f424e8e44df14edf49ce' and sha(splits)!='f125b312ab0e430d9213b9a935f80b9d98a6043056b32e4681b90bf35e24b0da':raise ValueError('El split no es la partición final congelada.')
    with splits.open(encoding='utf-8-sig',newline='') as f:rows=[r for r in csv.DictReader(f) if r['split']=='train']
    if len(rows)!=460 or len({r['name'] for r in rows})!=460:raise ValueError('Se requieren los460Train únicos del split final original.')
    resolved=[]
    for r in rows:
        path=(Path(images_dir)/r['name']) if images_dir else Path(r['image_path'])
        resolved.append((r,path))
    missing=[str(p) for _,p in resolved if not p.is_file()]
    evidence={'created_utc':datetime.now(timezone.utc).isoformat(),'split':'train','expected_cases':460,'splits_sha256':sha(splits),'available_images':460-len(missing),'missing_images':missing,'not_replaced_by_test_examples':True}
    if inventory_only:
        evidence['status']='inventory_only_not_band_statistics';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8');return
    if missing:raise FileNotFoundError(f'Faltan{len(missing)}imágenesTrain. No se estiman momentos con ejemplosTest ni min/max.')
    totals=Moments();zones=defaultdict(Moments);counts=defaultdict(int);inputs=[]
    for r,path in resolved:
        digest=sha(path)
        if digest!=r['source_sha256']:raise ValueError(f'Imagen no coincide con el hash original:{path}')
        with rasterio.open(path) as src:
            a=src.read()
            if a.shape!=(4,256,256) or src.dtypes!=('uint16',)*4:raise ValueError(f'FormatoTrain distinto:{path}')
            if (src.read_masks()==0).any() or (a==65535).any():raise ValueError(f'NoData enTrain:{path}')
            if str(src.crs)!='EPSG:32717' or src.res!=(10,10):raise ValueError(f'GrillaTrain inesperada:{path}')
        # The original loader cast to float32 and divided once, prior to the
        # model's own affine layer. Preserve exactly that convention.
        x=a.astype('float32')/np.float32(10000.)
        totals.add(x);zones[r['zone']].add(x);counts[r['zone']]+=1
        inputs.append({'name':r['name'],'zone':r['zone'],'sha256':digest})
    evidence.update(status='all_frozen_train_inputs_measured',units='physical reflectance as original DN/10000 loader',augmentation=False,resampling=False,clipping=False,statistical_convention='per-pixel descriptive population moments, ddof=0',rows=totals.result(),by_zone={z:{'n_cases':counts[z],'rows':mom.result()} for z,mom in zones.items()},inputs=inputs,model_affine_not_used_as_training_statistics=True)
    assert totals.n==460*256*256
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(f'EstadísticosTrain reales guardados:{out}; píxeles/banda:{totals.n}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--splits',required=True);p.add_argument('--images-dir');p.add_argument('--out',required=True);p.add_argument('--inventory-only',action='store_true');a=p.parse_args();run(a.splits,a.images_dir,a.out,a.inventory_only)
