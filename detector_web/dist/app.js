"use strict";
const $=id=>document.getElementById(id);
let busy=false, browserResult=null, selectedFile=null, activeController=null;
let comparisonApi=null;
const comparisonReady=import("./comparacion.js?v=4.0.0").then(module=>comparisonApi=module.createComparison({onExports:renderExports,status,setBusy,isBusy:()=>busy}));
const format=(value,digits=1)=>new Intl.NumberFormat("es-EC",{maximumFractionDigits:digits}).format(value);
function status(message,error=false){$("status").textContent=message;$("status").classList.toggle("error",error);$("status").setAttribute("role",error?"alert":"status");}
function badge(text,className=""){ $("result-badge").textContent=text;$("result-badge").className="result-badge "+className; }
function stage(current){for(const [i,name] of ["image","analysis","results"].entries()){const el=$("step-"+name);el.classList.toggle("current",i===current);el.classList.toggle("done",i<current);}}
function setBusy(value){busy=value;for(const name of ["analyze","image","scl","confirm-bands","clear","compare-references","ref-water","ref-cloud","ref-pond","ref-origin","ref-notes","ref-confirm"])$(name).disabled=value;$("progress").hidden=!value;$("cancel").hidden=!value;document.querySelector(".configuration").setAttribute("aria-busy",String(value));document.body.classList.toggle("is-busy",value);document.body.dataset.state=value?"processing":browserResult?"result":selectedFile?"ready":"reference";}
function releaseResult(){comparisonApi?.reset();document.body.dataset.state=busy?"processing":selectedFile?"ready":"reference";if(browserResult){for(const url of Object.values(browserResult.files))URL.revokeObjectURL(url);browserResult=null;}for(const name of ["rgb","classes"]){$(name).hidden=true;$(name).removeAttribute("src");}for(const name of ["empty-rgb","empty-classes"])$(name).hidden=false;for(const name of ["numbers","notes","exports"]){$(name).replaceChildren();$(name).hidden=true;}$("download").hidden=true;$("download").removeAttribute("href");$("layer-control").hidden=true;$("layer-legend").hidden=true;document.querySelector(".legend[aria-label=\"Leyenda\"]").hidden=false;$("layer").value="preview_classes.png";$("layer-description").textContent="Agua y fondo";$("result-title").textContent=selectedFile?"Tu próxima detección":"Dos vistas. Una misma escena.";$("result-subtitle").textContent=selectedFile?"Archivo seleccionado. Inicia el análisis para comparar las vistas.":"Carga un GeoTIFF para ver la imagen y su segmentación.";$("grid-info").textContent="Resolución de entrada · 10 m";badge(selectedFile?"Imagen seleccionada":"Sin análisis");stage(0);}
function imageChanged(file){if(busy)return;selectedFile=file||null;$("ref-confirm").checked=false;releaseResult();$("file-label").textContent=file?.name||".tif / .tiff · hasta 128 MiB";$("file-meta").textContent=file?format(file.size/1048576,2)+" MiB · seleccionado":"";$("file-meta").hidden=!file;status(file?"Imagen seleccionada. Comprueba las bandas antes de analizar.":"Listo para cargar una imagen.");}
$("scl").addEventListener("change",()=>{if(!busy){$("ref-confirm").checked=false;releaseResult();status("La calidad SCL cambió. Inicia un nuevo análisis para aplicar esta capa.");}});
$("image").addEventListener("change",()=>imageChanged($("image").files[0]));
for(const type of ["dragenter","dragover"])$("dropzone").addEventListener(type,event=>{event.preventDefault();if(!busy)$("dropzone").classList.add("drag-active");});
for(const type of ["dragleave","drop"])$("dropzone").addEventListener(type,event=>{event.preventDefault();$("dropzone").classList.remove("drag-active");});
$("dropzone").addEventListener("drop",event=>{if(busy)return;const files=event.dataTransfer?.files;if(files?.length!==1){status("Selecciona una sola imagen GeoTIFF.",true);return;}imageChanged(files[0]);});
$("layer").addEventListener("change",()=>{const name=$("layer").value;if(!/^(preview_(classes|overlay|candidates|water|background|cloud|ponds)|reference_(water|background|cloud|ponds)|difference_(water|background|cloud|ponds))\.png$/.test(name))return;if(browserResult)$("classes").src=browserResult.files[name];$("layer-description").textContent=$("layer").selectedOptions[0].textContent;
 const custom=name.startsWith("reference_")||name.startsWith("difference_")||/preview_(water|background|cloud|ponds)/.test(name),legend=$("layer-legend");document.querySelector('.legend[aria-label="Leyenda"]').hidden=custom;legend.hidden=!custom;legend.replaceChildren();
 if(custom){const diff=name.startsWith("difference_"),entries=diff?[["true-positive","Coincide la clase"],["false-positive","Sobreestimación"],["false-negative","Omisión"],["true-negative","Ausencia coincidente"],["invalid","No evaluado"]]:[[name.includes("ponds")?"pond":name.includes("cloud")?"cloud":name.includes("background")?"land":"water","Clase seleccionada"],["true-negative","Resto evaluable"],["invalid","Sin datos / no revisado"]];for(const [cls,title] of entries){const span=document.createElement("span"),i=document.createElement("i");i.className=cls;span.append(i,document.createTextNode(title));legend.append(span);}}
});
for(const href of ["#method","#privacy"])for(const link of document.querySelectorAll('a[href="'+href+'"]'))link.addEventListener("click",()=>document.querySelector(href).open=true);
$("clear").addEventListener("click",()=>{if(busy)return;selectedFile=null;releaseResult();$("upload-form").reset();$("file-label").textContent=".tif / .tiff · hasta 128 MiB";$("file-meta").hidden=true;status("Imagen y resultados liberados de esta página.");});
$("cancel").addEventListener("click",()=>{status("Cancelando el análisis…");activeController?.abort();comparisonApi?.cancel();});
async function run(task){
 if(busy)return;
 setBusy(true);releaseResult();badge("Analizando","working");stage(1);$("result-title").textContent="Analizando escena";$("result-subtitle").textContent="Preparando las bandas y la detección de agua.";$("progress").value=0;status("Preparando imagen…");activeController=new AbortController();
 let timeout;

 try{timeout=setTimeout(()=>activeController?.abort(new DOMException("Se alcanzó el tiempo máximo de análisis.","TimeoutError")),15*60*1000);activeController.signal.throwIfAborted();await task(activeController.signal);}
 catch(error){const aborted=activeController?.signal.aborted;if(aborted)releaseResult();badge(aborted?"Cancelado":"Revisa la imagen");stage(0);$("result-title").textContent=aborted?"Análisis cancelado":"Revisa tu escena";$("result-subtitle").textContent=selectedFile?.name||"Comprueba las bandas y la georreferencia del GeoTIFF.";status(aborted?"Análisis cancelado. Puedes cargar otra imagen.":String(error?.message||"No se pudo terminar el análisis."),!aborted);}
 finally{clearTimeout(timeout);activeController=null;setBusy(false);}
}
$("upload-form").addEventListener("submit",event=>{event.preventDefault();if(busy)return;const image=selectedFile||$("image").files[0];if(!image){status("Selecciona una imagen GeoTIFF para continuar.",true);$("image").focus();return;}run(signal=>beginBrowser(image,$("scl").files[0],signal));});
function showResult(data){
 const files = data.files;
 document.body.dataset.state="result";const report=data.report||{},stats=report.counts||{},valid=stats.classifiable_pixels??0,water=stats.water_pixels??0,total=stats.total_pixels??0;
 $("rgb").src=files["preview_rgb.png"];$("classes").src=files["preview_classes.png"];
 for(const name of ["rgb","classes"])$(name).hidden=false;for(const name of ["empty-rgb","empty-classes"])$(name).hidden=true;
 $("result-title").textContent="Detección terminada";$("result-subtitle").textContent=data.filename||"Imagen satelital";badge("Análisis completo","complete");stage(2);
 $("grid-info").textContent=report.input?.width+" × "+report.input?.height+" px · "+(total?format(valid/total*100,2):"0")+" % analizable";
 $("download").href=files["water_mask.tif"];$("download").download="water_mask.tif";$("download").textContent="Descargar máscara";$("download").hidden=false;
 $("layer-control").hidden=false;
 const cloud=report.input?.scl_available?stats.cloud_pixels:null,candidateCount=report.aquaculture_candidates?.count??0;
 $("numbers").replaceChildren();const entries=[[format(report.areas?.water_ha??water*.01,2)+" ha","Agua predicha"],[valid?format(water/valid*100,2)+" %":"—","Agua / área analizable"],[cloud==null?"Sin SCL":format(cloud,0),"Píxeles de nubes SCL"],[format(candidateCount,0),"Posibles estanques"]];
 for(const [value,label] of entries){const div=document.createElement("div");div.className="number";const strong=document.createElement("strong");strong.textContent=value;const span=document.createElement("span");span.textContent=label;div.append(strong,span);$("numbers").append(div);}$("numbers").hidden=false;
 const notes=[...(report.input?.warnings||[]),...(data.note?[data.note]:[])];if(total&&valid/total<.25)notes.unshift("Solo "+format(valid/total*100,2)+" % de la imagen es analizable. El área de agua no describe las zonas cubiertas o sin datos.");if(candidateCount)notes.push("Revisa cada candidato: la geometría no confirma una camaronera.");
 $("notes").replaceChildren();for(const note of notes){const p=document.createElement("p");p.textContent=typeof note==="string"?note:JSON.stringify(note);$("notes").append(p);}$("notes").hidden=!notes.length;
 renderExports(files);
}
function renderExports(files){$("exports").replaceChildren();const titles={"water_mask.tif":"Máscara GeoTIFF","probability.tif":"Probabilidad GeoTIFF","classes.tif":"Clases GeoTIFF","candidates.geojson":"Estanques GeoJSON","report.json":"Informe JSON","background_mask.tif":"Máscara de fondo","cloud_mask.tif":"Máscara de nubes SCL","ponds_mask.tif":"Agua en candidatos GeoTIFF"};for(const [name,title] of Object.entries(titles)){if(!files[name])continue;const a=document.createElement("a");a.href=files[name];a.download=name;a.textContent=title;$("exports").append(a);}$("exports").hidden=false;}
async function beginBrowser(image,scl,signal){
 const {processBrowser}=await import("./procesamiento-browser.js?v=4.0.0");
 const result=await processBrowser(image,{sclFile:scl,confirmed:$("confirm-bands").checked,signal,onProgress:(pct,message)=>{status(message);$("progress").value=pct;}});
 if(signal.aborted){Object.values(result.files).forEach(URL.revokeObjectURL);signal.throwIfAborted();}
 browserResult=result;showResult(result);await comparisonReady;await comparisonApi.show(result,{signal});if(comparisonApi.hasReferences())await comparisonApi.evaluate({internal:true,signal});signal.throwIfAborted();status("Análisis terminado. Los resultados están listos para descargar.");
}
function browserReady(){$("mode").textContent="Procesamiento en tu navegador";$("privacy-inline-text").textContent="Tus imágenes permanecen en tu equipo.";status("Listo para cargar una imagen. El modelo se cargará en el primer análisis.");}
browserReady();
window.addEventListener("pagehide",()=>{activeController?.abort();releaseResult();},{once:true});
