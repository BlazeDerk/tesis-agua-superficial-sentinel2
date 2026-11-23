// Límites aplicados antes de decodificar y reservar matrices de salida.
export const LIMITS=Object.freeze({pixels:4000000,opticalBytes:128*1024*1024,sclBytes:32*1024*1024,blockBytes:16*1024*1024,dimension:16384});
export function requireValid(ok,message){if(!ok)throw new Error(message);}
export function abortIfNeeded(signal){signal?.throwIfAborted();}
export async function validateFile(file,isScl=false,signal){
 abortIfNeeded(signal);requireValid(file&&typeof file.name==="string"&&/\.tiff?$/i.test(file.name),"Selecciona un archivo GeoTIFF .tif o .tiff.");
 const limit=isScl?LIMITS.sclBytes:LIMITS.opticalBytes;
 requireValid(Number.isSafeInteger(file.size)&&file.size>=16&&file.size<=limit,isScl?"SCL debe tener entre 16 bytes y 32 MiB.":"La imagen debe tener entre 16 bytes y 128 MiB. Recorta los mosaicos grandes antes de cargarlos.");
 requireValid(!isScl||!/clear_count|qc_/i.test(file.name),"QC_clear_count no es SCL y no identifica nubes.");
 const bytes=new Uint8Array(await file.slice(0,16).arrayBuffer());abortIfNeeded(signal);
 const little=bytes[0]===73&&bytes[1]===73,big=bytes[0]===77&&bytes[1]===77;
 requireValid(little||big,"La firma del archivo no corresponde a un TIFF.");
 const view=new DataView(bytes.buffer),magic=view.getUint16(2,little);
 requireValid(magic===42||(magic===43&&view.getUint16(4,little)===8&&view.getUint16(6,little)===0),"El encabezado TIFF está incompleto o no es compatible.");
}
export function validateDimensions(w,h,n,isScl=false){
 requireValid([w,h,n].every(Number.isSafeInteger)&&w>0&&h>0&&w<=LIMITS.dimension&&h<=LIMITS.dimension,"Las dimensiones de la imagen no son válidas.");
 requireValid(w*h<=LIMITS.pixels,"La imagen supera 4 millones de píxeles. Recórtala o utiliza el detector local para mosaicos mayores.");
 requireValid(isScl?n===1:n===4||n===5,isScl?"SCL debe tener una sola banda.":"Se requieren cuatro bandas ópticas y, como máximo, una banda adicional.");
}
export function validateGeo(scale,tie){
 requireValid(scale.length>=2&&tie.length>=6&&scale.every(Number.isFinite)&&tie.every(Number.isFinite),"La georreferencia contiene valores ausentes o no finitos.");
 requireValid(Math.abs(scale[0]-10)<1e-7&&Math.abs(scale[1]-10)<1e-7,"Se requiere una grilla GeoTIFF de 10 × 10 metros.");
 const origin=[tie[3]-tie[0]*10,tie[4]+tie[1]*10];
 requireValid(origin.every(Number.isFinite),"El origen de la imagen no es válido.");
 return origin;
}
export function validateBlock({w,h,n,bits,tileWidth,tileHeight,rowsPerStrip,planar=1}){
 requireValid(planar===1||planar===2,"La organización de bandas TIFF no es compatible.");
 const tw=tileWidth==null?w:Number(tileWidth),th=tileHeight==null?Math.min(h,Number(rowsPerStrip??h)):Number(tileHeight);
 requireValid(Number.isSafeInteger(tw)&&tw>0&&Number.isSafeInteger(th)&&th>0,"Las dimensiones del bloque TIFF no son válidas.");
 const bytes=tw*th*(planar===2?Math.max(...bits):bits.reduce((a,b)=>a+b,0))/8;
 requireValid(Number.isSafeInteger(bytes)&&bytes<=LIMITS.blockBytes,"El TIFF contiene bloques descomprimidos de más de 16 MiB. Reexporta como GeoTIFF en mosaico (tiled) para analizarlo.");
}

