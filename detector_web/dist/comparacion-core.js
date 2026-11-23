export const DEFINITIONS=Object.freeze({
 water:{title:"Agua",origin:"U-Net / VGG16",domain:"Superficie analizable (agua o fondo)"},
 background:{title:"Fondo sin agua",origin:"U-Net / VGG16",domain:"Superficie analizable (agua o fondo)"},
 cloud:{title:"Nubes",origin:"SCL · códigos 8/9/10",domain:"Entrada válida, incluidas nubes, sombras y nieve"},
 ponds:{title:"Agua en candidatos a estanque",origin:"Geometría + agua predicha",domain:"Superficie analizable; cajas aproximadas, sin confirmar uso camaronero"}
});
export function metricsFromCounts({tp,fp,fn,tn}){
 for(const n of [tp,fp,fn,tn])if(!Number.isSafeInteger(n)||n<0)throw new Error("Conteos de comparación inválidos.");
 const ratio=(a,b)=>b?a/b:null,total=tp+fp+fn+tn,union=tp+fp+fn;
 return{tp,fp,fn,tn,evaluated_pixels:total,reference_positive_pixels:tp+fn,predicted_positive_pixels:tp+fp,
  iou:ratio(tp,union),f1:ratio(2*tp,2*tp+fp+fn),precision:ratio(tp,tp+fp),sensitivity:ratio(tp,tp+fn),
  pixel_accuracy:ratio(tp+tn,total),undefined_reason:!total?"No hay píxeles evaluables.":!union?"La clase no aparece en predicción ni referencia.":null};
}
export function eligible(label,key){return key==="cloud"?label>0:label===1||label===2;}
export function positive(label,key,pondMask,index){return key==="water"?label===2:key==="background"?label===1:key==="cloud"?label===3:pondMask[index]===1;}
export function compareBinary(classes,reference,key,pondMask){
 if(!Object.hasOwn(DEFINITIONS,key))throw new Error("Capa de comparación desconocida.");
 if(classes.length!==reference.length)throw new Error("Dimensiones de referencia incompatibles.");
 let tp=0,fp=0,fn=0,tn=0,excluded=0;
 for(let i=0;i<classes.length;i++){
  const r=reference[i];if(r!==0&&r!==1&&r!==255&&r!==65535)throw new Error("La referencia solo admite 0, 1 y valores sin revisar 255/65535.");
  if(!eligible(classes[i],key)||r===255||r===65535){excluded++;continue;}
  const pred=positive(classes[i],key,pondMask,i),truth=key==="background"?r===0:r===1;
  if(pred&&truth)tp++;else if(pred)fp++;else if(truth)fn++;else tn++;
 }
 return{...metricsFromCounts({tp,fp,fn,tn}),excluded_pixels:excluded,scene_evaluated_fraction:classes.length?(tp+fp+fn+tn)/classes.length:0};
}
export function candidateFootprint(data){
 const {classes,w,h,preview_size,features}=data;
 if(!Number.isSafeInteger(w)||!Number.isSafeInteger(h)||w*h!==classes.length||classes.length>4000000)throw new Error("Grilla de comparación inválida.");
 const mask=new Uint8Array(classes.length),[pw,ph]=preview_size;
 if(!(pw>0&&ph>0)||features.length>250)throw new Error("Geometría candidata inválida.");
 for(const feature of features){
  const box=feature.properties?.bbox_preview_pixels;
  if(!Array.isArray(box)||box.length!==4||!box.every(Number.isFinite))throw new Error("Falta la caja de la geometría candidata.");
  const [x0,y0,x1,y1]=box;
  if(x0<0||y0<0||x1<=x0||y1<=y0||x1>pw||y1>ph)throw new Error("Caja candidata fuera de la imagen.");
  const left=Math.floor(x0*w/pw),right=Math.min(w,Math.ceil(x1*w/pw)),top=Math.floor(y0*h/ph),bottom=Math.min(h,Math.ceil(y1*h/ph));
  for(let y=top;y<bottom;y++)for(let x=left;x<right;x++){const i=y*w+x;if(classes[i]===2)mask[i]=1;}
 }
 return mask;
}
export function comparisonRows(report,data,pondMask){
 const c=report.counts,n=c.total_pixels;
 const pondPixels=pondMask?pondMask.reduce((a,b)=>a+b,0):null;
 return Object.entries(DEFINITIONS).map(([key,definition])=>{
  const available=key!=="cloud"||!!report.input.scl_available;
  const pixels=!available?null:key==="water"?c.water_pixels:key==="background"?c.non_water_pixels:key==="cloud"?c.cloud_pixels:pondPixels;
  return{key,...definition,available,pixels,area_ha:pixels===null?null:pixels/100,scene_fraction:pixels===null?null:pixels/n,
   status:!available?"Se requiere SCL":key==="ponds"?"Huella aproximada para revisión":"Disponible",metrics:null};
 });
}
export function safeCSV(rows){
 const cell=v=>{let s=v==null?"":String(v);if(typeof v==="string"&&/^[=+\-@\t\r]/.test(s))s="'"+s;return '"'+s.replace(/"/g,'""')+'"';};
 const header=["capa","origen","pixeles","area_ha","fraccion_escena","iou","f1","precision","sensibilidad","exactitud_pixeles","TP","FP","FN","TN","evaluados","positivos_referencia","referencia","tipo_referencia"];
 const data=rows.map(r=>[r.title,r.origin,r.pixels,r.area_ha,r.scene_fraction,r.metrics?.iou,r.metrics?.f1,r.metrics?.precision,r.metrics?.sensitivity,r.metrics?.pixel_accuracy,r.metrics?.tp,r.metrics?.fp,r.metrics?.fn,r.metrics?.tn,r.metrics?.evaluated_pixels,r.metrics?.reference_positive_pixels,r.reference?.filename,r.reference?.kind]);
 return "\ufeff"+[header,...data].map(row=>row.map(cell).join(",")).join("\r\n");
}

