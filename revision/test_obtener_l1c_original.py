import tempfile,unittest,zlib
from pathlib import Path
from obtener_l1c_original import write_deflate,selected_entries

class ExtractionTests(unittest.TestCase):
    def test_stream_extraction_and_crc(self):
        data=b'fixture original TIFF bytes'*2000
        z=zlib.compressobj(wbits=-15);compressed=z.compress(data)+z.flush()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'member';r=write_deflate([compressed[i:i+7] for i in range(0,len(compressed),7)],p,len(data),zlib.crc32(data)&0xffffffff)
            self.assertEqual(p.read_bytes(),data);self.assertEqual(r['file_bytes'],len(data))
            with self.assertRaises(ValueError):write_deflate([compressed],Path(d)/'bad',len(data),0)
            with self.assertRaises(ValueError):write_deflate([compressed],Path(d)/'oversize',5,0)

if __name__=='__main__':unittest.main()
