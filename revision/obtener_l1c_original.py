"""Extract only the three original L1C ZIP members. Default is a download plan.

Never requests a complete multi-gigabyte archive. Requires explicit --download,
bounded HTTP 206 ranges, and validates member size/CRC32. A whole-archive
checksum is not verified. Failed .partial files are retained for diagnosis.
"""
import argparse,hashlib,json,struct,urllib.request,zlib
from pathlib import Path

def selected_entries(index_root):
    found={}
    for path in Path(index_root).glob('part*.zip.central_directory.json'):
        archive=json.loads(path.read_text(encoding='utf-8'))
        for e in archive['entries']:
            if e['filename'] in {f'{s}/sentinel12_s2_{s}_img.tif' for s in [23,25,33]}:
                if e['filename'] in found:raise ValueError('Miembro repetido en el inventario.')
                found[e['filename']]={'index_file':str(path),'archive_url':archive['url'],'archive_size':archive['size'],**e}
    if len(found)!=3:raise ValueError('Faltan los tres miembros originales previstos.')
    return [found[k] for k in sorted(found)]

def request_range(url,total,start,length):
    end=start+length-1
    req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','User-Agent':'ScientificArticleReview/1.0','Accept-Encoding':'identity'})
    with urllib.request.urlopen(req,timeout=120) as r:
        if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{total}':
            raise ValueError('El servidor no respetó el rango; se aborta sin descargar el ZIP completo.')
        data=r.read(length+1)
    if len(data)!=length:raise ValueError('Longitud de rango inesperada.')
    return data

def write_deflate(chunks,target,expected_size,expected_crc):
    decoder=zlib.decompressobj(-15);size=0;crc=0;digest=hashlib.sha256()
    def write(out,block):
        nonlocal size,crc
        size+=len(block)
        if size>expected_size:raise ValueError('El miembro supera el tamaño publicado.')
        crc=zlib.crc32(block,crc);digest.update(block);out.write(block)
    with Path(target).open('xb') as out:
        for chunk in chunks:
            remaining=chunk
            while remaining:
                block=decoder.decompress(remaining,32*1024*1024)
                write(out,block);remaining=decoder.unconsumed_tail
            if decoder.unused_data:raise ValueError('Datos extra tras el flujo comprimido.')
        write(out,decoder.flush())
    if not decoder.eof or size!=expected_size or crc&0xffffffff!=expected_crc:
        raise ValueError('El miembro no supera las comprobaciones de final, tamaño y CRC32.')
    return {'file_bytes':size,'crc32':f'{crc&0xffffffff:08x}','sha256':digest.hexdigest()}

def main(a):
    entries=selected_entries(a.index_root)
    planned=sum(e['compress_size'] for e in entries)
    plan={'status':'download_plan_only','compressed_bytes':planned,'uncompressed_bytes':sum(e['file_size'] for e in entries),'source':'https://doi.org/10.5281/zenodo.11278238','members':entries}
    print(json.dumps(plan,ensure_ascii=False,indent=2))
    if not a.download:return
    if planned>a.max_gb*1_000_000_000:raise ValueError('Se supera --max-gb; no se inició la descarga.')
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True);receipts=[]
    for e in entries:
        if e['compress_type']!=8 or e['flag_bits']&1:raise ValueError('Se requiere miembro DEFLATE no cifrado.')
        target=out/e['filename'];target.parent.mkdir(parents=True,exist_ok=True)
        partial=target.with_suffix('.tif.partial')
        if target.exists() or partial.exists():raise ValueError(f'La salida ya existe: {target}; no se sobreescribe.')
        header=request_range(e['archive_url'],e['archive_size'],e['header_offset'],30)
        if header[:4]!=b'PK\x03\x04':raise ValueError('Cabecera ZIP local no válida.')
        name_len,extra_len=struct.unpack_from('<HH',header,26)
        name=request_range(e['archive_url'],e['archive_size'],e['header_offset']+30,name_len)
        if name.decode('utf-8')!=e['filename']:raise ValueError('Nombre local distinto del inventario central.')
        start=e['header_offset']+30+name_len+extra_len
        def chunks():
            done=0
            while done<e['compress_size']:
                length=min(8*1024*1024,e['compress_size']-done)
                data=request_range(e['archive_url'],e['archive_size'],start+done,length)
                done+=length
                print(f"{e['filename']}: {done}/{e['compress_size']} bytes comprimidos",flush=True)
                yield data
        fingerprint=write_deflate(chunks(),partial,e['file_size'],e['CRC'])
        partial.rename(target)
        receipts.append({'filename':e['filename'],'archive_url':e['archive_url'],'archive_whole_checksum_verified':False,**fingerprint})
        (out/'adquisicion_l1c.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index-root',default='insumos_externos/fuentes');p.add_argument('--out-dir',default='l1c_original');p.add_argument('--max-gb',type=float,default=6);p.add_argument('--download',action='store_true');main(p.parse_args())
