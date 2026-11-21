"""Extract only original public S2 reference+validity TIFFs via bounded ZIP ranges."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import zlib
import zipfile
from datetime import datetime,timezone
from consultar_fuentes import RangeFile

OUT = Path(__file__).resolve().parent
REFS = OUT/'referencias'
LIMIT = 100_000_000

class AuditedRangeFile(RangeFile):
    def __init__(self,*args):
        super().__init__(*args)
        self.ranges=[]
    def read(self,size=-1):
        start=self.pos
        raw=super().read(size)
        if raw:self.ranges.append({'offset':start,'length':len(raw)})
        return raw

def extract_archive(directory, chosen):
    remote=AuditedRangeFile(directory['url'],directory['size'])
    outputs=[]
    with zipfile.ZipFile(remote) as archive:
        for entry in chosen:
            key=entry['filename']
            target=REFS/key
            target.parent.mkdir(exist_ok=True,parents=True)
            if target.exists():
                raw=target.read_bytes()
                if len(raw)!=entry['file_size'] or (zlib.crc32(raw)&0xffffffff)!=entry['CRC']:
                    raise ValueError(f'Existing file failed size/CRC validation: {key}')
                outputs.append({'filename':key,'status':'existing_verified','file_bytes':len(raw),
                    'crc32':f'{entry["CRC"]:08x}','sha256':hashlib.sha256(raw).hexdigest(),
                    'archive_header_offset':entry['header_offset'],'archive_compress_size':entry['compress_size']})
                continue
            before=len(remote.ranges)
            content=bytearray()
            with archive.open(key) as stream:
                while True:
                    chunk=stream.read(900000)
                    if not chunk:break
                    content.extend(chunk)
                    if len(content)>20_000_000:
                        raise ValueError('Reference uncompressed size exceeded cap')
            assert len(content)==entry['file_size']
            assert zlib.crc32(content)&0xffffffff==entry['CRC']
            target.write_bytes(content)
            outputs.append({'filename':key,'status':'downloaded_verified','file_bytes':len(content),
                'crc32':f'{entry["CRC"]:08x}','sha256':hashlib.sha256(content).hexdigest(),
                'archive_header_offset':entry['header_offset'],'archive_compress_size':entry['compress_size'],
                'http_ranges':remote.ranges[before:]})
            target.with_suffix('.tif.acquisition.json').write_text(json.dumps({
                'archive':directory['archive'],'archive_url':directory['url'],**outputs[-1]},indent=2),encoding='utf-8')
            print('Verified',key,len(content),flush=True)
    return {'archive':directory['archive'],'archive_url':directory['url'],'archive_size':directory['size'],
        'archive_whole_checksum_verified':False,'http_bytes':remote.bytes,'http_requests':remote.calls,'outputs':outputs}

def main():
    metas={int(f.stem.split('_')[1]):json.loads(f.read_text()) for f in (OUT/'items').glob('*.json')}
    assert len(metas)==65
    test={key:item for key,item in metas.items() if item['properties']['split']=='test'}
    assert sorted(test)==[23,25,29,31,33,35,47,53,57,75,77,78,80,82,88,89]
    archive_jobs=[]
    for path in sorted(OUT.glob('*.central_directory.json')):
        directory=json.loads(path.read_text())
        chosen=[e for e in directory['entries'] if '_s2_' in e['filename']
            and e['filename'].endswith(('_msk.tif','_valid.tif'))
            and int(e['filename'].split('/')[0]) in test]
        if chosen:archive_jobs.append((directory,chosen))
    predicted=sum(e['compress_size'] for _,es in archive_jobs for e in es)
    if predicted>LIMIT:raise ValueError(f'Predicted download {predicted} exceeds cap {LIMIT}')
    freeze={'created_utc':datetime.now(timezone.utc).isoformat(),'source_record':'https://zenodo.org/records/11278238',
        'version':'v1.0.1','license':'CC-BY-4.0','all_scene_count':65,
        'official_split_counts':{'train':33,'val':16,'test':16},
        'selection':'All 16 official test scene IDs, selected from metadata before image/index/model evaluation',
        'reference_origin':'semi_automated_NDWI_initialization_then_multiple_manual_correction_iterations_by_three_experts',
        'reference_not_blind_to_NDWI':True,'s2_original_radiometry':'L1C_TOA_scaled_10000',
        's2_original_band_order':['B2','B3','B4','B8','B11','B12'],
        'water_mask_codes':{'0':'no_water','1':'water'},
        'validity_mask_codes':{'0':'invalid_cloud_shadow_nodata','1':'valid'},
        'metadata_geometry_warning':'STAC bbox/geometry numerical values appear projected; do not treat as WGS84. Use TIFF CRS/transform.',
        'predicted_compressed_reference_bytes':predicted,'download_limit_bytes':LIMIT,
        'test_scenes':[]}
    for key,item in sorted(test.items()):
        props=item['properties']
        source_ids=props['s2_srcids']
        freeze['test_scenes'].append({'id':key,'source_metadata':f'items/sentinel12_{key}_meta.json',
            'split':props['split'],'date_s2':props['date_s2'],'s2_source_ids':source_ids,
            'mgrs_tiles':[s.split('_T')[1].split('_')[0] for s in source_ids],
            'landcover':props['landcover'],'official_datetime_field_not_acquisition':props.get('datetime')})
    (OUT/'resumen.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Frozen all16 Test scene IDs; downloading {predicted} compressed reference bytes',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        results=list(pool.map(lambda x:extract_archive(*x),archive_jobs))
    final={'created_utc':datetime.now(timezone.utc).isoformat(),'archive_whole_checksums_verified':False,
        'individual_crc32_verified':True,'individual_sha256_computed':True,
        'http_bytes':sum(x['http_bytes'] for x in results),
        'files':sum(len(x['outputs']) for x in results),'archives':results}
    assert final['files']==32
    assert final['http_bytes']<LIMIT
    (OUT/'adquisicion_referencias.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in final.items() if k!='archives'}),flush=True)

if __name__=='__main__':main()
