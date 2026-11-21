"""Adapta referencias publicadas S1S2-Water a L2A de la misma adquisición.

No calcula índices ni consulta predicciones. Selección espacial determinista,
congelada antes de descargar la óptica. No descarga los ZIP de imágenes L1C.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent
API = 'https://earth-search.aws.element84.com/v1/search'
SALT = 's1s2water-l2a-v1'
SIZE = 256
PER_SCENE = 16
BANDS = [('B2', 'blue'), ('B3', 'green'), ('B4', 'red'), ('B8', 'nir'), ('B11', 'swir16')]
PRODUCT = re.compile(r'^(S2[AB])_MSIL[12][AC]_(\d{8}T\d{6})_N\d{4}_(R\d{3})_T(\d{2}[A-Z]{3})_')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def product_key(value):
    m = PRODUCT.match(value)
    if not m:
        raise ValueError(f'ID Sentinel-2 no reconocido: {value}')
    return m.groups()


def source_items():
    items = []
    for path in (ROOT / 'fuentes/items').glob('*.json'):
        item = read_json(path)
        if item['properties'].get('split') == 'test':
            assert len(item['properties']['s2_srcids']) == 1, path
            items.append((int(item['id'].split('_')[1]), path, item))
    items.sort(key=lambda x: x[0])
    assert len(items) == 16, f'Se esperaban los 16 casos Test oficiales, hay {len(items)}.'
    return items


def fetch_json(url):
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'TesisWaterExternalEvaluation/1.0'})
            with urllib.request.urlopen(req, timeout=45) as response:
                raw = response.read(8_000_001)
            if len(raw) > 8_000_000:
                raise ValueError('Respuesta STAC mayor que el límite de metadatos.')
            return json.loads(raw)
        except Exception as exc:
            last = exc
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f'No se pudo consultar {url}: {last}')


def catalog_one(info):
    scene, path, item = info
    source = item['properties']['s2_srcids'][0]
    platform, sensing, orbit, tile = product_key(source)
    date = sensing[:8]
    day_start = datetime.strptime(date, '%Y%m%d')
    interval = f"{day_start:%Y-%m-%d}T00:00:00Z/{day_start + timedelta(days=1):%Y-%m-%d}T00:00:00Z"
    query = {'mgrs:utm_zone': {'eq': int(tile[:2])}, 'mgrs:latitude_band': {'eq': tile[2]}, 'mgrs:grid_square': {'eq': tile[3:]}}
    params = {'collections': 'sentinel-2-l2a', 'datetime': interval, 'query': json.dumps(query), 'limit': 100}
    result = fetch_json(API + '?' + urllib.parse.urlencode(params))
    matches = []
    for feature in result.get('features', []):
        candidate = feature['properties'].get('s2:product_uri', '')
        if candidate and product_key(candidate) == (platform, sensing, orbit, tile):
            if all(key in feature['assets'] for _, key in BANDS):
                matches.append(feature)
    if not matches:
        raise ValueError(f'Sin contraparte L2A con identidad exacta para {scene}: {source}')
    # Rule fixed before predictions: oldest available L2A generation, then ID.
    matches.sort(key=lambda x: (x['properties'].get('s2:generation_time', '9999'), x['id']))
    chosen = matches[0]
    target = ROOT / 'fuentes/l2a' / f'{scene}.json'
    write_json(target, chosen)
    write_json(target.with_suffix('.consulta.json'), {
        'original_l1c': source, 'query': params,
        'selection_rule': 'earliest_s2_generation_time_then_item_id_among_exact_identity_matches',
        'matching_ids': [x['id'] for x in matches], 'chosen': chosen['id'],
    })
    return {'scene_id': str(scene), 'l1c_source': source, 'l2a_id': chosen['id'],
            'l2a_product': chosen['properties']['s2:product_uri'], 'same_acquisition_key': list(product_key(source)),
            'item_path': target.relative_to(ROOT).as_posix(), 'item_sha256': sha(target)}


def catalog():
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(catalog_one, info): info[0] for info in source_items()}
        for future in as_completed(futures):
            try:
                row = future.result()
                rows.append(row)
                print(f"L2A {row['scene_id']}: {row['l2a_id']}", flush=True)
            except Exception as exc:
                errors.append({'scene_id': futures[future], 'error': str(exc)})
    rows.sort(key=lambda x: int(x['scene_id']))
    write_json(ROOT / 'catalogo_l2a.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'matched': rows, 'errors': errors})
    if errors:
        raise RuntimeError(f'Hay {len(errors)} casos sin contraparte: véase catalogo_l2a.json.')


def ranked_windows(scene, height, width):
    candidates = []
    for row in range(0, height - SIZE + 1, SIZE):
        for col in range(0, width - SIZE + 1, SIZE):
            key = f'{SALT}|{scene}|r{row}|c{col}'
            rank = hashlib.sha256(key.encode('ascii')).hexdigest()
            candidates.append((rank, row, col))
    assert len(candidates) >= PER_SCENE
    return sorted(candidates)[:PER_SCENE]


def plan():
    target = ROOT / 'plan_congelado.json'
    if target.exists():
        print('El plan ya está congelado; no se sobrescribe.', flush=True)
        return
    catalog_data = read_json(ROOT / 'catalogo_l2a.json')
    assert not catalog_data['errors'] and len(catalog_data['matched']) == 16
    scenes, cases = [], []
    by_id = {x['scene_id']: x for x in catalog_data['matched']}
    for scene, source_path, item in source_items():
        scene = str(scene)
        mask = ROOT / f'fuentes/referencias/{scene}/sentinel12_s2_{scene}_msk.tif'
        valid = mask.with_name(f'sentinel12_s2_{scene}_valid.tif')
        with rasterio.open(mask) as ref, rasterio.open(valid) as quality:
            assert ref.crs and ref.shape == quality.shape and ref.transform == quality.transform and ref.crs == quality.crs
            assert ref.count == quality.count == 1
            assert tuple(ref.res) == (10.0, 10.0)
            windows = ranked_windows(scene, ref.height, ref.width)
            matched = by_id[scene]
            # Distinct tiles from the same overpass share a conservative cluster.
            platform, sensing, orbit, tile = product_key(matched['l1c_source'])
            cluster = f'{platform}_{sensing}_{orbit}'
            scenes.append({**matched, 'cluster_id': cluster, 'mask_path': mask.relative_to(ROOT).as_posix(),
                           'valid_path': valid.relative_to(ROOT).as_posix(), 'mask_sha256': sha(mask), 'valid_sha256': sha(valid),
                           'source_item_sha256': sha(source_path), 'crs': str(ref.crs), 'transform': list(ref.transform),
                           'shape': list(ref.shape), 'full_grid_windows': (ref.height // SIZE) * (ref.width // SIZE),
                           'discarded_edge_rows': ref.height % SIZE, 'discarded_edge_cols': ref.width % SIZE})
            for rank, row, col in windows:
                cases.append({'case_id': f'S{int(scene):03d}_r{row:05d}_c{col:05d}', 'scene_id': scene,
                              'cluster_id': cluster, 'row_off': row, 'col_off': col, 'height': SIZE, 'width': SIZE, 'rank_sha256': rank})
    write_json(target, {
        'version': '1.0.0', 'created_utc': datetime.now(timezone.utc).isoformat(), 'split': 'test',
        'salt': SALT, 'selection': 'lowest_SHA256_of_salt_scene_row_col_on_nonoverlapping_full_256_grid',
        'cases_per_scene': PER_SCENE, 'selection_does_not_use_water_cloud_or_predictions': True,
        'reference_origin': 'public_expert_refined_ndwi',
        'reference_limit': 'NDWI-initialized masks iteratively refined by experts; not fully independent of NDWI',
        'input_adaptation': 'Same acquisition L2A counterpart; not original S1S2-Water L1C benchmark',
        'scenes': scenes, 'cases': cases, 'no_unet_inference_yet': True,
    })
    print(f'Plan congelado: {len(scenes)} escenas, {len(cases)} ventanas. SHA256 {sha(target)}', flush=True)


def dump_tiff(path, array, profile, descriptions, nodata):
    profile = dict(profile)
    profile.update(driver='GTiff', count=array.shape[0], dtype=str(array.dtype), nodata=nodata,
                   compress='deflate', tiled=True, blockxsize=SIZE, blockysize=SIZE)
    with rasterio.open(path, 'w', **profile) as out:
        out.write(array)
        out.descriptions = tuple(descriptions)
        out.update_tags(source='S1S2-Water references paired with same-acquisition Sentinel-2 L2A',
                        radiometry='surface_reflectance_scale_offset_applied_once' if array.shape[0] == 5 else 'categorical')


def download_scene(scene_info, cases, plan_hash):
    scene = scene_info['scene_id']
    item_path = ROOT / scene_info['item_path']
    assert sha(item_path) == scene_info['item_sha256']
    item = read_json(item_path)
    for key, digest in [('mask_path', 'mask_sha256'), ('valid_path', 'valid_sha256')]:
        assert sha(ROOT / scene_info[key]) == scene_info[digest]
    directory = ROOT / 'datos' / f'S{int(scene):03d}'
    directory.mkdir(parents=True, exist_ok=True)
    done = []
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
                      GDAL_HTTP_TIMEOUT='60', GDAL_HTTP_MAX_RETRY='3', GDAL_HTTP_RETRY_DELAY='2',
                      GDAL_CACHEMAX=64_000_000, VSI_CACHE='FALSE'), ExitStack() as stack:
        ref = stack.enter_context(rasterio.open(ROOT / scene_info['mask_path']))
        valid = stack.enter_context(rasterio.open(ROOT / scene_info['valid_path']))
        sources = []
        band_meta = []
        for name, key in BANDS:
            asset = item['assets'][key]
            meta = asset['raster:bands'][0]
            assert 'scale' in meta and 'offset' in meta
            scale, offset = float(meta['scale']), float(meta['offset'])
            ds = stack.enter_context(rasterio.open(asset['href']))
            assert ds.count == 1 and ds.crs == ref.crs, (scene, name, ds.crs, ref.crs)
            if name != 'B11':
                assert ds.res == (10, 10)
                source = stack.enter_context(WarpedVRT(ds, crs=ref.crs, transform=ref.transform,
                                                       width=ref.width, height=ref.height, dtype='float32',
                                                       resampling=Resampling.nearest, nodata=65535.0))
            else:
                assert ds.res == (20, 20)
                source = stack.enter_context(WarpedVRT(ds, crs=ref.crs, transform=ref.transform,
                                                       width=ref.width, height=ref.height, dtype='float32',
                                                       resampling=Resampling.bilinear, nodata=65535.0))
            sources.append((source, scale, offset))
            band_meta.append({'band': name, 'href': asset['href'], 'scale': scale, 'offset': offset,
                              'native_resolution': list(ds.res), 'source_nodata': ds.nodata,
                              'native_shape': list(ds.shape), 'native_transform': list(ds.transform),
                              'resampling': 'bilinear' if name == 'B11' else 'nearest'})
        for i, case in enumerate(cases, 1):
            case_id = case['case_id']
            stamp = directory / f'{case_id}.json'
            if stamp.exists():
                saved = read_json(stamp)
                assert saved['plan_sha256'] == plan_hash
                for key in ['optical', 'reference', 'validity']:
                    assert sha(ROOT / saved['manifest'][f'{key}_path']) == saved['manifest'][f'{key}_sha256']
                done.append(saved['manifest'])
                continue
            window = Window(case['col_off'], case['row_off'], SIZE, SIZE)
            reflectance = []
            for source, scale, offset in sources:
                raw = source.read(1, window=window, masked=True).astype('float32')
                physical = raw * np.float32(scale) + np.float32(offset)
                reflectance.append(physical.filled(np.nan))
            optical = np.stack(reflectance).astype('float32')
            reference = ref.read(1, window=window)
            quality = valid.read(1, window=window)
            assert reference.shape == quality.shape == (SIZE, SIZE)
            assert np.all(np.isin(reference, [0, 1])) and np.all(np.isin(quality, [0, 1]))
            profile = {'crs': ref.crs, 'transform': ref.window_transform(window), 'height': SIZE, 'width': SIZE}
            files = {
                'optical': directory / f'{case_id}_L2A.tif',
                'reference': directory / f'{case_id}_referencia.tif',
                'validity': directory / f'{case_id}_validez.tif',
            }
            dump_tiff(files['optical'], optical, profile, [b for b, _ in BANDS], float('nan'))
            dump_tiff(files['reference'], reference[None].astype('uint8'), profile, ['water_reference'], 255)
            dump_tiff(files['validity'], quality[None].astype('uint8'), profile, ['manual_validity'], 255)
            record = {k: case[k] for k in ['case_id', 'scene_id', 'cluster_id', 'row_off', 'col_off', 'height', 'width']}
            record.update(split='test', reference_origin='public_expert_refined_ndwi',
                          l1c_source=scene_info['l1c_source'], l2a_product=scene_info['l2a_product'])
            for key, path in files.items():
                record[f'{key}_path'] = path.relative_to(ROOT).as_posix()
                record[f'{key}_sha256'] = sha(path)
            write_json(stamp, {'plan_sha256': plan_hash, 'l2a_item_sha256': scene_info['item_sha256'],
                              'source_bands': band_meta, 'manifest': record,
                              'acquired_utc': datetime.now(timezone.utc).isoformat(),
                              'b11_resampling': 'bilinear_20m_to_reference_10m',
                              'visible_nir_alignment': 'nearest_neighbor_to_reference_10m_grid',
                              'reference_and_quality': 'unchanged categorical windows, no resampling',
                              'no_indices_or_predictions_generated': True})
            done.append(record)
            print(f'Datos {scene}: {i}/{len(cases)} ({case_id})', flush=True)
    return done


def download(workers, scene_limit):
    frozen = ROOT / 'plan_congelado.json'
    plan_data = read_json(frozen)
    plan_hash = sha(frozen)
    by_scene = {s['scene_id']: [c for c in plan_data['cases'] if c['scene_id'] == s['scene_id']] for s in plan_data['scenes']}
    scenes = plan_data['scenes'][:scene_limit] if scene_limit else plan_data['scenes']
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(download_scene, s, by_scene[s['scene_id']], plan_hash): s['scene_id'] for s in scenes}
        for future in as_completed(jobs):
            try:
                rows.extend(future.result())
            except Exception as exc:
                errors.append({'scene_id': jobs[future], 'error': str(exc)})
                print(f'ERROR {jobs[future]}: {exc}', flush=True)
    rows.sort(key=lambda x: (int(x['scene_id']), x['row_off'], x['col_off']))
    complete = not errors and len(rows) == len(plan_data['cases'])
    target = ROOT / ('manifest_s1s2.csv' if complete else 'manifest_parcial_NO_EVALUAR.csv')
    if rows:
        with target.open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(ROOT / 'estado_adquisicion.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': plan_hash,
        'expected_scenes': len(plan_data['scenes']), 'expected_cases': len(plan_data['cases']),
        'completed_cases_this_run': len(rows), 'complete': complete, 'errors': errors,
        'manifest': target.name if rows else None, 'manifest_sha256': sha(target) if rows else None,
        'unet_evaluated': False, 'spectral_indices_evaluated': False,
    })
    print(f'Descarga completa={complete}; casos={len(rows)}; errores={len(errors)}', flush=True)
    if errors:
        raise RuntimeError('Hay descargas pendientes: véase estado_adquisicion.json. Se permite reanudar.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['catalogo', 'plan', 'descargar'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--scene-limit', type=int, default=0, help='Solo QA de adquisición; no genera el manifiesto completo.')
    args = parser.parse_args()
    if args.action == 'catalogo':
        catalog()
    elif args.action == 'plan':
        plan()
    else:
        download(args.workers, args.scene_limit)
