import {DEFINITIONS,eligible,positive,compareBinary,candidateFootprint,comparisonRows,safeCSV} from "./comparacion-core.js?v=4.0.0";
import {validateFile,validateDimensions,validateGeo,validateBlock,abortIfNeeded,requireValid} from "./validacion.js?v=2.0.0";
const $=id=>document.getElementById(id),format=(v,d=2)=>new Intl.NumberFormat("es-EC",{maximumFractionDigits:d}).format(v);
const pc=v=>v==null?"—":format(v*100)+" %",NODATA=65535;
const REF_IDS={water:"ref-water",cloud:"ref-cloud",ponds:"ref-pond"};
const PALETTE={water:[22,129,189],background:[172,155,119],cloud:[242,246,251],ponds:[245,137,38]};
async function hash(file){const digest=await crypto.subtle.digest("SHA-256",await file.arrayBuffer());return Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,"0")).join("");}
export function validateReferenceNoData(value){requireValid(value!==0&&value!==1,"La referencia declara 0 o 1 como Sin datos y perdería etiquetas válidas. Reexpórtala sin NoData o usando 255/65535.");requireValid(value==null||value===255||value===65535,"Usa 255 o 65535 para las zonas no revisadas de la referencia.");}
async function reference(file,data,signal){
 requireValid(file.size<=32*1024*1024,"Cada referencia admite un máximo de 32 MiB.");
 await validateFile(file,false,signal);
 const tiff=await GeoTIFF.fromBlob(file,signal);
 try{
  const image=await tiff.getImage(),fd=image.fileDirectory,tag=async name=>fd.hasTag(name)?await fd.loadValue(name):null;
  requireValid(image.getSamplesPerPixel()===1,"La referencia debe tener una sola banda binaria.");
  validateDimensions(image.getWidth(),image.getHeight(),1,true);
  const bits=Array.from(await tag("BitsPerSample")||[8]),types=Array.from(await tag("SampleFormat")||[1]);
  requireValid(bits.length===1&&[8,16].includes(bits[0])&&types.every(v=>v===1),"La referencia requiere uint8 o uint16.");
  requireValid(fd.hasTag("TileOffsets")!==fd.hasTag("StripOffsets"),"La referencia tiene bloques ambiguos.");
  requireValid(!(fd.hasTag("StripOffsets")&&(fd.hasTag("TileWidth")||fd.hasTag("TileLength"))),"La referencia mezcla tiras y bloques.");
  const bw=image.getTileWidth(),bh=image.getTileHeight();
  validateBlock({w:image.getWidth(),h:image.getHeight(),n:1,bits,tileWidth:bw,tileHeight:bh,planar:1});
  const tiled=fd.hasTag("TileOffsets"),offsets=Array.from(await tag(tiled?"TileOffsets":"StripOffsets")||[],Number),sizes=Array.from(await tag(tiled?"TileByteCounts":"StripByteCounts")||[],Number);
  requireValid(offsets.length===Math.ceil(image.getWidth()/bw)*Math.ceil(image.getHeight()/bh)&&sizes.length===offsets.length&&offsets.every((v,i)=>Number.isSafeInteger(v)&&v>=0&&Number.isSafeInteger(sizes[i])&&sizes[i]>0&&v+sizes[i]<=file.size),"La referencia está truncada o tiene una tabla TIFF inválida.");
  const keys=image.getGeoKeys();
  requireValid(keys.ProjectedCSTypeGeoKey===32717&&keys.GTRasterTypeGeoKey!==2&&!fd.hasTag("ModelTransformation"),"La referencia debe usar EPSG:32717 y píxeles de área sin rotación.");
  const origin=validateGeo(Array.from(await tag("ModelPixelScale")||[]),Array.from(await tag("ModelTiepoint")||[]));
  requireValid(image.getWidth()===data.w&&image.getHeight()===data.h&&origin.every((v,i)=>Math.abs(v-data.origin[i])<1e-6),"La referencia no coincide con la extensión, tamaño y grilla de la escena.");
  const nodata=await image.getGDALNoData();validateReferenceNoData(nodata);
  const values=new Uint8Array(data.w*data.h).fill(255);
  for(let row=0;row<data.h;row+=256)for(let col=0;col<data.w;col+=256){
   abortIfNeeded(signal);const w=Math.min(256,data.w-col),h=Math.min(256,data.h-row);
   const raster=await image.readRasters({signal,window:[col,row,col+w,row+h],samples:[0],interleave:true});
   for(let y=0;y<h;y++)for(let x=0;x<w;x++){const v=raster[y*w+x];requireValid(v===0||v===1||v===255||v===65535,"La referencia contiene etiquetas distintas de 0/1/255/65535.");values[(row+y)*data.w+col+x]=v===0||v===1?v:255;}
  }
  return{values,filename:file.name,sha256:await hash(file),kind:$("ref-origin").value,declared_source:$("ref-notes").value.trim(),same_scene_confirmed:$("ref-confirm").checked};
 }finally{try{await tiff.close();}catch{}}
}
function meta(data,nodata=65535){return{width:data.w,height:data.h,ModelPixelScale:[10,10,0],ModelTiepoint:[0,0,0,...data.origin,0],GTModelTypeGeoKey:1,GTRasterTypeGeoKey:1,ProjectedCSTypeGeoKey:32717,ProjLinearUnitsGeoKey:9001,SamplesPerPixel:1,BitsPerSample:[16],SampleFormat:[1],GDAL_NODATA:String(nodata)};}
async function png(data,key,pondMask,ref=null,difference=false){
 const resize=Math.max(1,Math.max(data.w,data.h)/1280),w=Math.ceil(data.w/resize),h=Math.ceil(data.h/resize),bytes=new Uint8ClampedArray(w*h*4);
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){
  const i=Math.min(data.h-1,Math.floor(y*data.h/h))*data.w+Math.min(data.w-1,Math.floor(x*data.w/w)),p=(y*w+x)*4,label=data.classes[i],pred=positive(label,key,pondMask,i),valid=eligible(label,key);
  let color=[31,41,55];
  if(ref){const known=ref[i]!==255;if(known){const truth=key==="background"?ref[i]===0:ref[i]===1;color=difference?(valid?(pred&&truth?[72,201,135]:pred?[240,162,74]:truth?[233,114,173]:[69,92,109]):[31,41,55]):truth?PALETTE[key]:[69,92,109];}}
  else if(valid)color=pred?PALETTE[key]:[69,92,109];
  bytes.set([...color,255],p);
 }
 const canvas=document.createElement("canvas");canvas.width=w;canvas.height=h;canvas.getContext("2d").putImageData(new ImageData(bytes,w,h),0,0);
 return await new Promise((resolve,reject)=>canvas.toBlob(v=>v?resolve(v):reject(new Error("No se pudo generar la vista comparativa.")),"image/png"));
}
export function createComparison({onExports,status,setBusy,isBusy}){
 let result=null,rows=[],pondMask=null,derived=new Set(),controller=null,generation=0;
 const announce=(message,error=false)=>{$("comparison-status").textContent=message;$("comparison-status").classList.toggle("error",error);};
 const add=(name,blob)=>{if(result.files[name])URL.revokeObjectURL(result.files[name]);result.files[name]=URL.createObjectURL(blob);derived.add(name);return result.files[name];};
 const remove=name=>{if(result?.files[name]){URL.revokeObjectURL(result.files[name]);delete result.files[name];}derived.delete(name);};
 function documents(){
  if(!result)return;
  result.report.comparison={version:"1.0",model_sha256:result.report.model.sha256,threshold:result.report.model.threshold,rows,definitions:DEFINITIONS,source_note:"Agua/fondo de U-Net; nubes desde SCL; candidatos geométricos. Las métricas son concordancia con referencias declaradas.",candidate_definition:"Unión de rectángulos aproximados intersectada con agua predicha; no lindero ni uso camaronero confirmado.",pixel_area_ha:.01};
  add("comparison.json",new Blob([JSON.stringify(result.report.comparison,null,2)],{type:"application/json"}));
  add("comparison.csv",new Blob([safeCSV(rows)],{type:"text/csv;charset=utf-8"}));
  if(result.files["report.json"])URL.revokeObjectURL(result.files["report.json"]);
  result.files["report.json"]=URL.createObjectURL(new Blob([JSON.stringify(result.report,null,2)],{type:"application/json"}));derived.add("report.json");
  $("comparison-downloads").replaceChildren();
  for(const [name,title] of [["comparison.csv","Tabla CSV"],["comparison.json","Comparación JSON"],["reference_template.tif","Plantilla GeoTIFF para etiquetar"]]){
   if(!result.files[name])continue;const a=document.createElement("a");a.href=result.files[name];a.download=name;a.textContent=title;$("comparison-downloads").append(a);
  }$("comparison-downloads").hidden=false;onExports(result.files);
 }
 function draw(){
  $("comparison-rows").replaceChildren();$("metrics-details").replaceChildren();
  for(const row of rows){
   const tr=document.createElement("tr"),values=[row.title,row.origin,row.pixels==null?"—":format(row.pixels,0),row.area_ha==null?"—":format(row.area_ha),pc(row.scene_fraction),pc(row.metrics?.iou),pc(row.metrics?.f1)];
   for(const [i,value] of values.entries()){const td=document.createElement("td");td.textContent=value;if(i>=5)td.className="metric";tr.append(td);}
   $("comparison-rows").append(tr);
   if(row.metrics){const d=document.createElement("div");d.className="metric-detail";const strong=document.createElement("strong");strong.textContent=row.title;const a=document.createElement("p");a.textContent="Precisión: "+pc(row.metrics.precision)+" · Sensibilidad: "+pc(row.metrics.sensitivity)+" · Exactitud de píxeles: "+pc(row.metrics.pixel_accuracy);const b=document.createElement("p");b.textContent="TP "+row.metrics.tp+" · FP "+row.metrics.fp+" · FN "+row.metrics.fn+" · TN "+row.metrics.tn+" · "+format(row.metrics.evaluated_pixels,0)+" píxeles evaluados ("+pc(row.metrics.scene_evaluated_fraction)+" de la escena).";const c=document.createElement("p");c.textContent="Referencia: "+row.reference.filename+" · "+row.reference.kind+" · "+row.domain+(row.metrics.undefined_reason?" · "+row.metrics.undefined_reason:"");d.append(strong,a,b,c);$("metrics-details").append(d);}
  }$("metrics-details").hidden=!rows.some(r=>r.metrics);documents();
 }
 function option(name,title,disabled=false){let el=Array.from($("layer").options).find(o=>o.value===name);if(!el){el=document.createElement("option");el.value=name;$("layer").append(el);}el.textContent=title;el.disabled=disabled;}
 function invalidate(){
  const selectedReference=/^(reference_|difference_)/.test($("layer").value);
  if(!result)return;controller?.abort();for(const name of Array.from(derived))if(name.endsWith(".png")&&(name.startsWith("reference_")||name.startsWith("difference_")))remove(name);
  rows.forEach(r=>{r.metrics=null;delete r.reference;});for(const o of Array.from($("layer").options))if(o.value.startsWith("reference_")||o.value.startsWith("difference_"))o.remove();
  if(selectedReference||!result.files[$("layer").value]){$("layer").value="preview_classes.png";$("layer").dispatchEvent(new Event("change"));}
  $("comparison-error-legend").hidden=true;announce("Las referencias cambiaron. Recalcula las métricas para esta escena.");draw();
 }
 for(const id of [...Object.values(REF_IDS),"ref-origin","ref-notes","ref-confirm"])$(id).addEventListener("change",invalidate);
 async function show(value,{signal=null}={}){
  abortIfNeeded(signal);
  reset();result=value;rows=comparisonRows(value.report,value.data,null);$("comparison-panel").hidden=false;
  if(!value.data){draw();announce("La comparación por referencias y las capas individuales están disponibles en el modo navegador. Usa ?motor=navegador en la versión local.");return;}
  pondMask=candidateFootprint(value.data);rows=comparisonRows(value.report,value.data,pondMask);const c=value.report.counts;$("comparison-scope").textContent="Las capas conservan su origen. El agua en candidatos está incluida en el agua predicha. Área analizable: "+pc(c.classifiable_pixels/c.total_pixels)+". Además de las nubes, se excluyen del análisis de agua "+format(c.shadow_pixels/100)+" ha de sombra, "+format(c.snow_pixels/100)+" ha de nieve y "+format(c.invalid_pixels/100)+" ha sin datos.";
  for(const key of ["water","background","cloud","ponds"]){
   if(key==="cloud"&&!value.report.input.scl_available){option("preview_cloud.png","Solo nubes · requiere SCL",true);continue;}
   const name="preview_"+key+".png",preview=await png(value.data,key,pondMask);abortIfNeeded(signal);add(name,preview);option(name,"Solo "+(key==="water"?"agua":key==="background"?"fondo":key==="cloud"?"nubes SCL":"agua en candidatos"));
   if(key!=="water"){
    const array=new Uint16Array(value.data.classes.length).fill(NODATA);for(let i=0;i<array.length;i++)if(eligible(value.data.classes[i],key))array[i]=positive(value.data.classes[i],key,pondMask,i)?1:0;
    const buffer=await GeoTIFF.writeArrayBuffer(array,meta(value.data));abortIfNeeded(signal);add(key+"_mask.tif",new Blob([buffer],{type:"image/tiff"}));
   }
  }
  const blank=new Uint16Array(value.data.classes.length).fill(NODATA),template=await GeoTIFF.writeArrayBuffer(blank,meta(value.data));abortIfNeeded(signal);add("reference_template.tif",new Blob([template],{type:"image/tiff"}));
  announce("Sin referencias: se comparan áreas y capas. IoU y F1 requieren etiquetas de referencia.");
  draw();
 }
 async function evaluate({internal=false,signal=null}={}){
  if(!result?.data){announce("Primero analiza una imagen en el modo navegador.",true);return;}
  if(!internal&&isBusy())return;
  if(!Object.values(REF_IDS).some(id=>$(id).files.length)){announce("Añade al menos una referencia de agua, nubes o estanques.",true);return;}
  if(!$("ref-confirm").checked){$("reference-options").open=true;announce("Confirma que las referencias corresponden a la misma escena.",true);return;}
  abortIfNeeded(signal);if(!internal)setBusy(true);controller=new AbortController();const stop=()=>controller?.abort();signal?.addEventListener("abort",stop,{once:true});
  const current=++generation;let timeout;
  try{
   invalidate();announce("Comprobando referencias y comparando píxeles…");
   // invalidate aborts an older run; reserve the controller only after invalidating.
   controller=new AbortController();timeout=setTimeout(()=>controller?.abort(),120000);const ownSignal=controller.signal;
   for(const [key,id] of Object.entries(REF_IDS)){
    const file=$(id).files[0];if(!file)continue;if(key==="cloud"&&!result.report.input.scl_available){announce("Se necesita SCL para comparar una predicción de nubes.",true);continue;}
    const ref=await reference(file,result.data,ownSignal);abortIfNeeded(ownSignal);if(current!==generation)throw new DOMException("Comparación cancelada.","AbortError");
    const targets=key==="water"?["water","background"]:[key];
    for(const target of targets){const row=rows.find(r=>r.key===target);row.metrics=compareBinary(result.data.classes,ref.values,target,pondMask);row.reference={filename:ref.filename,sha256:ref.sha256,kind:ref.kind,declared_source:ref.declared_source,same_scene_confirmed:ref.same_scene_confirmed};const referencePng=await png(result.data,target,pondMask,ref.values);abortIfNeeded(ownSignal);add("reference_"+target+".png",referencePng);const differencePng=await png(result.data,target,pondMask,ref.values,true);abortIfNeeded(ownSignal);add("difference_"+target+".png",differencePng);option("reference_"+target+".png","Referencia · "+row.title);option("difference_"+target+".png","Omisiones y sobreestimaciones · "+row.title);}
   }
   abortIfNeeded(ownSignal);draw();$("comparison-error-legend").hidden=!rows.some(r=>r.metrics);if(!rows.some(r=>r.metrics)){announce("No se calcularon métricas: falta una predicción compatible. Las nubes necesitan SCL.",true);return;}
   const basis=$("ref-origin").value;announce(basis==="manual"?"Comparación con etiquetas manuales declaradas. Revisa el soporte y la cobertura evaluada.":basis==="derived"?"Prueba técnica: las referencias derivadas de esta predicción no miden exactitud independiente.":"Concordancia con referencias "+(basis==="automatic"?"automáticas":"de origen no especificado")+". Estos valores no sustituyen una validación de campo.");
  }catch(error){invalidate();announce(error.name==="AbortError"?"Comparación cancelada.":error.message,true);}
  finally{clearTimeout(timeout);signal?.removeEventListener("abort",stop);controller=null;if(!internal){setBusy(false);status("La detección sigue disponible. Revisa el estado de la comparación.");}}
 }
 $("compare-references").addEventListener("click",()=>evaluate());
 function reset(){
  controller?.abort();generation++;for(const name of Array.from(derived))remove(name);result=null;pondMask=null;rows=[];derived.clear();$("comparison-panel").hidden=true;$("metrics-details").replaceChildren();$("comparison-rows").replaceChildren();$("comparison-downloads").replaceChildren();$("comparison-error-legend").hidden=true;
  for(const o of Array.from($("layer").options))if(!["preview_classes.png","preview_overlay.png","preview_candidates.png"].includes(o.value))o.remove();
 }
 return{show,evaluate,reset,cancel:()=>controller?.abort(),hasReferences:()=>Object.values(REF_IDS).some(id=>$(id).files.length)};
}

