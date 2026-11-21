"""Fetch only public metadata and ZIP central directories, never whole archives."""
from pathlib import Path
from io import RawIOBase
import json
import urllib.request
import urllib.error
import zipfile
import hashlib
import time

OUT = Path(__file__).resolve().parent
MAX_READ = 1_000_000
HEADERS = {'User-Agent': 'ScientificArticleReview/1.0'}

def fetch(url, limit=1_000_000):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=35) as response:
        data = response.read(limit+1)
    if len(data) > limit:
        raise ValueError('Response exceeded metadata limit')
    return data

class RangeFile(RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.calls, self.bytes = 0, 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        if self.pos < 0: raise ValueError('negative seek')
        return self.pos
    def read(self, size=-1):
        if size < 0: size = self.size-self.pos
        size = min(size, self.size-self.pos)
        if size == 0: return b''
        if size > MAX_READ: raise ValueError(f'Range request too large: {size}')
        req = urllib.request.Request(self.url, headers={**HEADERS, 'Range': f'bytes={self.pos}-{self.pos+size-1}'})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=35) as response:
                    if response.status != 206:
                        raise ValueError(f'Server ignored Range ({response.status}); aborted before archive download')
                    expected=f'bytes {self.pos}-{self.pos+size-1}/{self.size}'
                    if response.headers.get('Content-Range') != expected:
                        raise ValueError(f'Unexpected Content-Range: {response.headers.get("Content-Range")}')
                    data=response.read(size+1)
                break
            except urllib.error.HTTPError as error:
                if error.code not in (429,500,502,503,504) or attempt==5:raise
                wait=min(45,10*(attempt+1))
                print(f'HTTP {error.code}, retry in {wait}s',flush=True)
                time.sleep(wait)
        if len(data) != size: raise ValueError('Incorrect range length')
        self.pos += len(data)
        self.calls += 1
        self.bytes += len(data)
        return data

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sources = {
      'zenodo_record_11278238.json':'https://zenodo.org/api/records/11278238',
      'catalog_autores.json':'https://raw.githubusercontent.com/MWieland/s1s2_water/main/data/catalog.json',
      'settings_autores.toml':'https://raw.githubusercontent.com/MWieland/s1s2_water/main/settings.toml',
      'split_autores.py':'https://raw.githubusercontent.com/MWieland/s1s2_water/main/prepare/split.py',
      'hf_torchgeo_tree.json':'https://huggingface.co/api/datasets/torchgeo/s1s2water/tree/main?recursive=true&limit=1000',
    }
    audit=[]
    for name,url in sources.items():
        raw=fetch(url)
        (OUT/name).write_bytes(raw)
        audit.append({'file':name,'url':url,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    record=json.loads((OUT/'zenodo_record_11278238.json').read_bytes())
    directories=[]
    for file in sorted(record['files'],key=lambda x:x['key']):
        if not file['key'].endswith('.zip'): continue
        remote=RangeFile(file['links']['self'],file['size'])
        try:
            with zipfile.ZipFile(remote) as z:
                entries=[{'filename':i.filename,'compress_type':i.compress_type,'compress_size':i.compress_size,
                    'file_size':i.file_size,'CRC':i.CRC,'header_offset':i.header_offset,'flag_bits':i.flag_bits}
                    for i in z.infolist()]
                metadata=[]
                for info in z.infolist():
                    if info.filename.endswith('_meta.json') and info.file_size<100000:
                        raw=z.read(info)
                        basename=Path(info.filename).name
                        (OUT/'items').mkdir(exist_ok=True)
                        (OUT/'items'/basename).write_bytes(raw)
                        metadata.append(basename)
            result={'archive':file['key'],'url':file['links']['self'],'size':file['size'],
                'entries':entries,'metadata_files':metadata,'http_range_requests':remote.calls,'http_bytes':remote.bytes}
            (OUT/(file['key']+'.central_directory.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
            directories.append({'archive':file['key'],'entries':len(entries),'methods':sorted(set(x['compress_type'] for x in entries)),
                'metadata':len(metadata),'range_requests':remote.calls,'bytes':remote.bytes})
            print(json.dumps(directories[-1]),flush=True)
        except Exception as e:
            directories.append({'archive':file['key'],'error':repr(e),'range_requests':remote.calls,'bytes':remote.bytes})
            print(json.dumps(directories[-1]),flush=True)
    (OUT/'auditoria_adquisicion_fuentes.json').write_text(json.dumps({'sources':audit,'directories':directories},indent=2),encoding='utf-8')

if __name__=='__main__':main()
