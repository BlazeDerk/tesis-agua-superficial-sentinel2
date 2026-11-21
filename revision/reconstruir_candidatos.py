"""Consulta actual de productos L2A candidatos sobre las tres geometrías de la v1 ejecutada."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request
import time

BASE = Path(__file__).resolve().parent
OUT = BASE / "catalogo_candidatos_recalculado"
SEARCH = "https://earth-search.aws.element84.com/v1/search"
ZONES = {"golfo_guayaquil": [-80.20,-2.82,-79.80,-2.38],
         "lagos_imbabura": [-78.47,0.10,-78.02,0.48], "rio_napo":[-77.80,-1.14,-77.42,-0.92]}
COUNTS_GEE = {"golfo_guayaquil":1414, "lagos_imbabura":417, "rio_napo":601}

def page(body):
    req=urllib.request.Request(SEARCH,data=json.dumps(body).encode(),
                               headers={"Content-Type":"application/json","User-Agent":"Scientific-reproducibility-Ecuador"})
    for retry in range(4):
        try:
            with urllib.request.urlopen(req,timeout=45) as response:
                return response.read()
        except OSError:
            if retry==3: raise
            time.sleep(1+2**retry)

def main():
    OUT.mkdir(exist_ok=False)
    start=datetime.now(timezone.utc).isoformat()
    records=[]; stages=[]
    for zone, bbox in ZONES.items():
        original={"collections":["sentinel-2-c1-l2a"],"bbox":bbox,
                  "datetime":"2020-01-01T00:00:00Z/2023-12-31T23:59:59.999Z","limit":100,
                  "query":{"eo:cloud_cover":{"lt":90}},
                  "fields":{"include":["id","collection","properties.datetime","properties.s2:product_uri",
                                       "properties.eo:cloud_cover","properties.platform","properties.s2:datatake_id",
                                       "properties.s2:processing_baseline"],"exclude":["assets","geometry","bbox","links"]}}
        current=original; ids=set(); receipts=[]
        while current:
            raw=page(current); j=json.loads(raw)
            receipts.append({"request":current,"response_sha256":hashlib.sha256(raw).hexdigest(),
                             "returned":len(j["features"])})
            for f in j["features"]:
                p=f["properties"]
                assert f["id"] not in ids and p["eo:cloud_cover"]<90
                ids.add(f["id"])
                records.append({"sector":zone, "collection_stac":f["collection"],"stac_id":f["id"],
                       "product_id":p["s2:product_uri"],"system_index_gee_original":"","datetime_utc":p["datetime"],
                       "cloud_percent_catalog":p["eo:cloud_cover"],"datatake":p.get("s2:datatake_id",""),
                       "processing_baseline":p.get("s2:processing_baseline",""),
                       "matched_historical_cloud_probability_join":"desconocido",
                       "historical_pixel_contribution":"desconocida"})
            nxt=next((l for l in j["links"] if l["rel"]=="next"),None)
            if nxt: assert nxt["href"]==SEARCH and nxt["method"]=="POST"
            current=(nxt["body"] if not nxt.get("merge") else {**current,**nxt["body"]}) if nxt else None
        assert len(ids)==j["numberMatched"]
        stages.append({"zone":zone,"stac_candidates":len(ids),"historical_gee_matched_count":COUNTS_GEE[zone],
                       "pages":receipts})
    with (OUT/"productos_candidatos_L2A.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=list(records[0])); w.writeheader();w.writerows(records)
    (OUT/"consulta_y_alcance.json").write_text(json.dumps({"started_utc":start,
         "finished_utc":datetime.now(timezone.utc).isoformat(),"url":SEARCH,"stages":stages,
         "interpretation":"Consulta de candidatos a Earth Search en la fecha de revisión. Colección C1 reprocesada, huella y cobertura de nubes del catálogo actual. El identificador de gránulo histórico GEE, el cruce original por system:index con probabilidades de nubes y la contribución por píxel se conservan como desconocidos.",
         "historical_filter":"GEE S2_SR_HARMONIZED y S2_CLOUD_PROBABILITY cruzados por system:index. Nubes de escena <90%, probabilidad <40%, exclusión SCL y bordes, dilatación 60 m, mediana y mínimo de tres observaciones por píxel.",
         },ensure_ascii=False,indent=2),encoding="utf-8")
    print([(s["zone"],s["stac_candidates"]) for s in stages])
if __name__=="__main__":
    main()
