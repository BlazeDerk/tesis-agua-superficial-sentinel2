// Inferencia local en el navegador para la versión privada alojada.
import {loadModel,predictDN,EXPECTED_MODEL_SHA256} from './engine.js?v=2.0.0';
const PALETTE=[[31,41,55],[172,155,119],[22,129,189],[242,246,251],[128,105,157],[182,231,242]];
import {validateFile,validateDimensions,validateGeo,validateBlock,abortIfNeeded} from './validacion.js?v=2.0.0';
let modelPromise=null;
const check=(ok,message)=>{if(!ok)throw new Error(message);};
const libraryPromises=new Map();
const LIBRARIES=[
 ['vendor/tf-4.22.0.min.js','tf','sha256-oWoWHXCTbmvNio1DgOG/fHJO3UYTWwrKULtASO0g3Zg='],
 ['vendor/geotiff-3.0.5.js','GeoTIFF','sha256-jPwB/8lFkEdm2PuJ8Z2nr7eZ40KDRFT1WS+88Lz+LbU='],
 ['vendor/proj4-2.22.0.js','proj4','sha256-r332U9kepZHzPSb7lYmQu9MHGy22RKTtq6RBzJhhpHQ=']
];
function awaitWithAbort(promise,signal){
 if(!signal)return promise;
 abortIfNeeded(signal);
 return new Promise((resolve,reject)=>{
   const abort=()=>reject(signal.reason||new DOMException('Análisis cancelado.','AbortError'));
   signal.addEventListener('abort',abort,{once:true});
   promise.then(resolve,reject).finally(()=>signal.removeEventListener('abort',abort));
 });
}
const libraryReady=name=>name==='tf'?typeof globalThis.tf?.tensor4d==='function'&&typeof globalThis.tf?.image?.resizeBilinear==='function':name==='GeoTIFF'?typeof globalThis.GeoTIFF?.fromBlob==='function':typeof globalThis.proj4==='function';
async function script(src,globalName,integrity){
 if(libraryReady(globalName))return;
 if(!libraryPromises.has(src))libraryPromises.set(src,new Promise((resolve,reject)=>{
   const el=document.createElement('script');el.src=src;el.integrity=integrity;
   let timer;
   const fail=()=>{clearTimeout(timer);el.remove();libraryPromises.delete(src);if(!libraryReady(globalName))delete globalThis[globalName];reject(new Error('No se pudo cargar una biblioteca verificada. Vuelve a intentarlo.'));};
   el.onload=()=>{clearTimeout(timer);libraryReady(globalName)?resolve():fail();};
   el.onerror=fail;timer=setTimeout(fail,30000);document.head.append(el);
 }));
 await libraryPromises.get(src);
}
async function libraries(signal){for(const lib of LIBRARIES){await awaitWithAbort(script(...lib),signal);}check(typeof globalThis.tf?.tensor4d==='function'&&typeof globalThis.tf?.image?.resizeBilinear==='function','El motor TensorFlow no se inicializó correctamente. Recarga la página.');}
async function getModel(onProgress,signal){
 await libraries(signal);abortIfNeeded(signal);
 if(!modelPromise)modelPromise=loadModel(new URL('model/manifest.json',location.href).href,{onProgress,signal});
 try{return await modelPromise;}catch(error){modelPromise=null;throw error;}
}
const normal=s=>String(s||'').trim().toUpperCase().replace(/^B0/,'B');
async function inspect(file,confirmed,isScl=false,signal){
  await validateFile(file,isScl,signal);
  const tiff=await GeoTIFF.fromBlob(file,signal),image=await tiff.getImage();
  abortIfNeeded(signal);
  const w=image.getWidth(),h=image.getHeight(),n=image.getSamplesPerPixel(),fd=image.fileDirectory,keys=image.getGeoKeys();
  validateDimensions(w,h,n,isScl);
  const tag=async name=>fd.hasTag(name)?await fd.loadValue(name):null;
  const bits=Array.from(await tag('BitsPerSample')||[8]);const sampleFormats=Array.from(await tag('SampleFormat')||[1]);
  if(bits.length===1&&n>1)while(bits.length<n)bits.push(bits[0]);
  check(bits.length===n&&[1,n].includes(sampleFormats.length),'Los metadatos de bandas TIFF son inconsistentes.');
  const tiled=fd.hasTag('TileOffsets'),stripped=fd.hasTag('StripOffsets');
  check(tiled!==stripped,'El TIFF tiene una organización de bloques ambigua o incompleta.');
  check(!(stripped&&(fd.hasTag('TileWidth')||fd.hasTag('TileLength'))),'El TIFF mezcla bloques y tiras de forma incompatible.');
  const blockWidth=image.getTileWidth(),blockHeight=image.getTileHeight(),planar=Number(await tag('PlanarConfiguration')||1);
  validateBlock({w,h,n,bits,tileWidth:blockWidth,tileHeight:blockHeight,planar});
  const offsets=Array.from(await tag(tiled?'TileOffsets':'StripOffsets')||[],Number),byteCounts=Array.from(await tag(tiled?'TileByteCounts':'StripByteCounts')||[],Number);
  const blockCount=Math.ceil(w/blockWidth)*Math.ceil(h/blockHeight)*(planar===2?n:1);
  check(offsets.length===blockCount&&byteCounts.length===blockCount,'La tabla de bloques TIFF está incompleta.');
  check(offsets.every((offset,i)=>Number.isSafeInteger(offset)&&offset>=0&&Number.isSafeInteger(byteCounts[i])&&byteCounts[i]>0&&offset+byteCounts[i]<=file.size),'El TIFF está truncado o contiene posiciones de bloques inválidas.');
  check(bits.every(v=>isScl?[8,16].includes(v):v===16)&&sampleFormats.every(v=>v===1),'Se requiere reflectancia uint16; JPG/PNG, RGB de 8 bits y reflectancia float no son compatibles.');
  check(keys.GTRasterTypeGeoKey!==2,'Reexporta la imagen con píxeles de área (PixelIsArea) para preservar su grilla.');
  check(keys.ProjectedCSTypeGeoKey===32717,'La versión de navegador requiere EPSG:32717, el sistema de la tesis.');
  check(!fd.hasTag('ModelTransformation'),'Reexporta una grilla sin rotación a 10 metros.');
  const scale=Array.from(await tag('ModelPixelScale')||[]),tie=Array.from(await tag('ModelTiepoint')||[]);
  const origin=validateGeo(scale,tie);
  let indexes=[0,1,2,3];const descriptions=[];
  for(let c=0;c<n;c++){const meta=await image.getGDALMetadata(c);descriptions.push(normal(meta?.DESCRIPTION));}
  if(isScl){check(n===1,'SCL debe contener una sola banda.');check(!descriptions.some(d=>/CLEAR_COUNT|^QC/.test(d)),'Esta capa de calidad no es SCL.');indexes=[0];}
  else if(['B2','B3','B4','B8'].every(b=>descriptions.filter(d=>d===b).length===1)){indexes=['B2','B3','B4','B8'].map(b=>descriptions.indexOf(b));}
  else{check((n===4||n===5)&&!descriptions.some(d=>['B2','B3','B4','B8','RED','GREEN','BLUE','ALPHA'].includes(d)),'No se identifican inequívocamente las cuatro bandas Sentinel-2.');check(confirmed,'Confirma el orden B2, B3, B4, B8 si el archivo no tiene nombres de bandas.');}
  let nodata;try{nodata=await image.getGDALNoData();}catch{nodata=null;}
  return{image,tiff,w,h,n,indexes,descriptions,origin,scale,tie,nodata,filename:file.name};
}
function canvasBlob(canvas){return new Promise((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(new Error('No se pudo generar la vista de la imagen.')),'image/png'));}
function png(bytes,w,h){const canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;const ctx=canvas.getContext('2d');check(ctx,'No se pudo crear una vista en este navegador.');ctx.putImageData(new ImageData(bytes,w,h),0,0);return canvasBlob(canvas);}
function stretch(values){const sorted=Array.from(values).sort((a,b)=>a-b);return sorted.length?[sorted[Math.floor(sorted.length*.02)],sorted[Math.min(sorted.length-1,Math.floor(sorted.length*.98))]]:[0,1];}
function candidates(labels,w,h,source){
  const seen=new Uint8Array(labels.length),queue=new Int32Array(labels.length),results=[];let examined=0;
  const sx=source.w/w*10,sy=source.h/h*10;
  for(let seed=0;seed<labels.length;seed++){
    if(seen[seed]||labels[seed]!==2)continue;
    let head=0,tail=1,minX=w,minY=h,maxX=0,maxY=0;queue[0]=seed;seen[seed]=1;
    while(head<tail){const pos=queue[head++],x=pos%w,y=Math.floor(pos/w);minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);
      const near=[];if(x>0)near.push(pos-1);if(x<w-1)near.push(pos+1);if(y>0)near.push(pos-w);if(y<h-1)near.push(pos+w);
      for(const other of near)if(!seen[other]&&labels[other]===2){seen[other]=1;queue[tail++]=other;}
    }
    examined++;const area=tail*sx*sy,box=(maxX-minX+1)*(maxY-minY+1),fill=tail/box,aspect=Math.max((maxX-minX+1)*sx,(maxY-minY+1)*sy)/Math.min((maxX-minX+1)*sx,(maxY-minY+1)*sy);
    if(area<2500||area>500000||fill<.60||aspect>5||minX===0||minY===0||maxX===w-1||maxY===h-1)continue;
    const boxPixels=[minX,minY,maxX+1,maxY+1];
    const ring=[[minX,minY],[maxX+1,minY],[maxX+1,maxY+1],[minX,maxY+1],[minX,minY]].map(([x,y])=>proj4('+proj=utm +zone=17 +south +datum=WGS84 +units=m +no_defs','EPSG:4326',[source.origin[0]+x*sx,source.origin[1]-y*sy]));
    results.push({type:'Feature',properties:{id:results.length+1,confirmed_aquaculture:false,estimated_water_area_ha:area/10000,bbox_fill:fill,aspect_ratio:aspect,geometry_represents:'rectángulo aproximado para revisión, no lindero del estanque',bbox_preview_pixels:boxPixels},geometry:{type:'Polygon',coordinates:[ring]}});
    if(results.length>=250)break;
  }
  return{features:results,examined};
}
function georef(source,dtype){return{width:source.w,height:source.h,ModelPixelScale:[10,10,0],ModelTiepoint:[0,0,0,...source.origin,0],GTModelTypeGeoKey:1,GTRasterTypeGeoKey:1,ProjectedCSTypeGeoKey:32717,ProjLinearUnitsGeoKey:9001,SamplesPerPixel:1,BitsPerSample:[dtype==='float32'?32:16],SampleFormat:[dtype==='float32'?3:1],GDAL_NODATA:dtype==='float32'?'-9999':'65535'};}
export async function processBrowser(file,{sclFile,confirmed=false,onProgress=()=>{},signal}={}){
  const started=performance.now();await libraries(signal);onProgress(1,'Comprobando la imagen…');const src=await inspect(file,confirmed,false,signal),scl=sclFile?await inspect(sclFile,true,true,signal):null;
  if(scl)check(src.w===scl.w&&src.h===scl.h&&src.origin.every((v,i)=>Math.abs(v-scl.origin[i])<1e-6),'SCL debe tener el mismo tamaño y ubicación que la imagen. Usa el exportador de la tesis para alinearlas.');
  abortIfNeeded(signal);
  const files={};
  try{
  const total=Math.ceil(src.w/256)*Math.ceil(src.h/256),probability=new Float32Array(src.w*src.h).fill(-9999),mask=new Uint16Array(src.w*src.h).fill(65535),classes=new Uint8Array(src.w*src.h);
  const counts={total_pixels:src.w*src.h,valid_input_pixels:0,classifiable_pixels:0,water_pixels:0,non_water_pixels:0,cloud_pixels:0,shadow_pixels:0,snow_pixels:0,invalid_pixels:0};
  // Validar SCL antes de gastar recursos en inferencia.
  if(scl)for(let row=0;row<src.h;row+=256)for(let col=0;col<src.w;col+=256){const values=await scl.image.readRasters({signal,window:[col,row,Math.min(col+256,src.w),Math.min(row+256,src.h)],samples:[0],interleave:true});check(values.every(v=>v<=11||v===65535||v===scl.nodata),'SCL contiene valores ajenos a los códigos 0–11.');}
  const model=await getModel(progress=>onProgress(2,'Cargando modelo de agua en el navegador…'),signal);let done=0;
  for(let row=0;row<src.h;row+=256)for(let col=0;col<src.w;col+=256){
    abortIfNeeded(signal);
    const width=Math.min(256,src.w-col),height=Math.min(256,src.h-row),window=[col,row,col+width,row+height];const optical=await src.image.readRasters({signal,window,samples:src.indexes,interleave:true}),quality=scl?await scl.image.readRasters({signal,window,samples:[0],interleave:true}):null;
    const tensor=new Uint16Array(256*256*4),labels=new Uint8Array(width*height);let predictable=0;
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){const p=y*width+x;let valid=true,allZero=true;for(let c=0;c<4;c++){const v=optical[p*4+c];if(v===65535||(src.nodata!=null&&v===src.nodata))valid=false;if(v!==0)allZero=false;}valid=valid&&!allZero;const code=quality?.[p];if(quality&&(code===0||code===1||code===65535||code===scl.nodata))valid=false;
      let label=0;if(valid){counts.valid_input_pixels++;label=quality&&[8,9,10].includes(code)?3:quality&&code===3?4:quality&&code===11?5:1;if(label===3)counts.cloud_pixels++;else if(label===4)counts.shadow_pixels++;else if(label===5)counts.snow_pixels++;else{counts.classifiable_pixels++;predictable++;}}
      labels[p]=label;
    }
    for(let y=0;y<256;y++)for(let x=0;x<256;x++){const p=Math.min(y,height-1)*width+Math.min(x,width-1);for(let c=0;c<4;c++)tensor[(y*256+x)*4+c]=labels[p]===1?optical[p*4+c]:0;}
    const values=predictable?(await predictDN(model,tensor,{signal})).data:null;
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){const p=y*width+x,index=(row+y)*src.w+col+x;let label=labels[p];if(label===1){const v=values[y*256+x];check(Number.isFinite(v)&&v>=0&&v<=1,'El modelo produjo probabilidades inválidas.');probability[index]=v;mask[index]=v>=.5?1:0;if(mask[index]){label=2;counts.water_pixels++;}else counts.non_water_pixels++;}classes[index]=label;}
    done++;onProgress(5+done/total*85,`Analizando bloque ${done} de ${total}…`);await new Promise(resolve=>setTimeout(resolve,0));
  }
  counts.invalid_pixels=counts.total_pixels-counts.valid_input_pixels;check(counts.valid_input_pixels>0,'No hay píxeles válidos para analizar.');
  abortIfNeeded(signal);
  onProgress(93,'Preparando las vistas y los archivos de salida…');
  const resize=Math.max(1,Math.max(src.w,src.h)/1280),pw=Math.max(1,Math.ceil(src.w/resize)),ph=Math.max(1,Math.ceil(src.h/resize));
  const visible=await src.image.readRasters({signal,samples:[src.indexes[2],src.indexes[1],src.indexes[0]],width:pw,height:ph,interleave:true,resampleMethod:'nearest'}),previewLabels=new Uint8Array(pw*ph),ranges=[];
  for(let y=0;y<ph;y++)for(let x=0;x<pw;x++)previewLabels[y*pw+x]=classes[Math.min(src.h-1,Math.floor(y*src.h/ph))*src.w+Math.min(src.w-1,Math.floor(x*src.w/pw))];
  for(let c=0;c<3;c++){const vals=[];for(let p=0;p<previewLabels.length;p++)if(previewLabels[p])vals.push(visible[p*3+c]);ranges.push(stretch(vals));}
  const rgb=new Uint8ClampedArray(pw*ph*4),classRGBA=new Uint8ClampedArray(rgb.length),overlay=new Uint8ClampedArray(rgb.length);
  for(let p=0;p<previewLabels.length;p++){const label=previewLabels[p];for(let c=0;c<3;c++){const [low,high]=ranges[c];rgb[p*4+c]=label?Math.max(0,Math.min(255,Math.floor((visible[p*3+c]-low)/Math.max(1,high-low)*255))):PALETTE[0][c];classRGBA[p*4+c]=PALETTE[label][c];overlay[p*4+c]=label?Math.floor(rgb[p*4+c]*.48+PALETTE[label][c]*.52):PALETTE[0][c];}rgb[p*4+3]=classRGBA[p*4+3]=overlay[p*4+3]=255;}
  const greenNir=await src.image.readRasters({signal,samples:[src.indexes[1],src.indexes[3]],width:pw,height:ph,interleave:true,resampleMethod:'nearest'}),candidateLabels=new Uint8Array(previewLabels.length);
  for(let p=0;p<candidateLabels.length;p++)if(previewLabels[p]===2&&greenNir[p*2]>greenNir[p*2+1])candidateLabels[p]=2;
  const aquaculture=candidates(candidateLabels,pw,ph,src),candidateCanvas=document.createElement('canvas');candidateCanvas.width=pw;candidateCanvas.height=ph;const context=candidateCanvas.getContext('2d');context.putImageData(new ImageData(overlay,pw,ph),0,0);context.strokeStyle='rgb(245,137,38)';context.lineWidth=3;for(const feature of aquaculture.features){const [x0,y0,x1,y1]=feature.properties.bbox_preview_pixels;context.strokeRect(x0,y0,x1-x0,y1-y0);}
  const warnings=[];if(!scl)warnings.push('No hay SCL: no se identifican nubes. El fondo puede incluir nubes, sombras, vegetación o construcciones.');if(src.n>4)warnings.push('Las bandas adicionales se ignoraron y nunca entraron al modelo.');warnings.push('Los candidatos combinan agua predicha con NDWI > 0 para separar diques; son indicios geométricos, no camaroneras confirmadas.');
  const report={status:'completed',version:'2.0.0-browser',created_utc:new Date().toISOString(),input:{width:src.w,height:src.h,crs:'EPSG:32717',transform:[10,0,src.origin[0],0,-10,src.origin[1]],optical_indexes:src.indexes.map(v=>v+1),scl_available:!!scl,filename:file.name,warnings},model:{sha256:EXPECTED_MODEL_SHA256,threshold:.5,trained_classes:['agua','fondo'],probabilities_calibrated:false},counts,areas:{water_ha:counts.water_pixels/100,non_water_ha:counts.non_water_pixels/100,cloud_ha:counts.cloud_pixels/100},aquaculture_candidates:{count:aquaculture.features.length,is_validated_class:false,method:'Componentes de agua de 0,25–50ha; rectángulos aproximados, relleno≥0,60, relación≤5, sin tocar borde.',analysis_resolution_m:[src.w/pw*10,src.h/ph*10]},runtime:{mode:'browser',elapsed_seconds:(performance.now()-started)/1000,tiles:total,backend:model.backend},limitations:['MNDWI son pseudoetiquetas; no prueba física independiente.','El desempeño en agua escasa de Imbabura necesita más referencias.','Nubes solo desde SCL; geometría de estanque no confirma actividad camaronera.','Bloques de256 sin solapamiento: pueden aparecer discontinuidades en bordes.']};
  const add=(name,blob)=>files[name]=URL.createObjectURL(blob);
  add('preview_rgb.png',await png(rgb,pw,ph));add('preview_classes.png',await png(classRGBA,pw,ph));add('preview_overlay.png',await png(overlay,pw,ph));add('preview_candidates.png',await canvasBlob(candidateCanvas));
  const exported=[];
  for(const [name,array,dtype,nodata] of [['water_mask.tif',mask,'uint16',65535],['classes.tif',new Uint16Array(classes),'uint16',0],['probability.tif',probability,'float32',-9999]]){
    abortIfNeeded(signal);
    const buffer=await GeoTIFF.writeArrayBuffer(array,{...georef(src,dtype),GDAL_NODATA:String(nodata)});
    const reopened=await(await GeoTIFF.fromArrayBuffer(buffer)).getImage();
    check(reopened.getWidth()===src.w&&reopened.getHeight()===src.h&&reopened.getGeoKeys().ProjectedCSTypeGeoKey===32717,'No se pudo conservar la georreferencia del resultado.');
    const outputOrigin=await reopened.getOrigin();check(outputOrigin[0]===src.origin[0]&&outputOrigin[1]===src.origin[1],'Cambió la ubicación al exportar.');
    const sample=await reopened.readRasters({signal,window:[0,0,Math.min(16,src.w),1],samples:[0],interleave:true});check(sample.every((v,i)=>v===array[i]),'La exportación alteró los valores del resultado.');
    check(await reopened.getGDALNoData()===nodata,'No se conservó el valor Sin datos al exportar.');
    exported.push({file:name,width:src.w,height:src.h,crs:'EPSG:32717',nodata,readback_first_row_pixels:sample.length,values_preserved:true});
    add(name,new Blob([buffer],{type:'image/tiff'}));
  }
  report.export_verification={method:'GeoTIFF exportado y reabierto en navegador; grilla y muestra primera fila comparadas.',files:exported};
  add('candidates.geojson',new Blob([JSON.stringify({type:'FeatureCollection',features:aquaculture.features},null,2)],{type:'application/geo+json'}));add('report.json',new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
  abortIfNeeded(signal);
  onProgress(100,'Análisis terminado.');return{report,files,filename:file.name,data:{classes,w:src.w,h:src.h,origin:src.origin,preview_size:[pw,ph],features:aquaculture.features}};
  }catch(error){Object.values(files).forEach(URL.revokeObjectURL);throw error;}
  finally{try{await src.tiff.close();}catch{}if(scl)try{await scl.tiff.close();}catch{}}
}
