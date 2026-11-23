import fs from 'node:fs';
import path from 'node:path';
const root=process.cwd(),dist=path.join(root,'dist'),client=path.join(dist,'client'),server=path.join(dist,'server');
fs.mkdirSync(client,{recursive:true});fs.mkdirSync(server,{recursive:true});
for(const name of fs.readdirSync(dist)){if(['client','server','.openai','_headers','index.html'].includes(name))continue;fs.cpSync(path.join(dist,name),path.join(client,name),{recursive:true});}
const html=fs.readFileSync(path.join(dist,'index.html'),'utf8');
const csp="default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; worker-src 'none'; frame-ancestors 'self' https://chatgpt.com";
const workerSource=`const HTML=${JSON.stringify(html)};
const CSP=${JSON.stringify(csp)};
const ALLOWED=new Set(['app.js','estilos.css','engine.js','validacion.js','procesamiento-browser.js','comparacion.js','comparacion-core.js','model/manifest.json','vendor/tf-4.22.0.min.js','vendor/geotiff-3.0.5.js','vendor/proj4-2.22.0.js','vendor/LICENSE-tensorflowjs.txt','vendor/LICENSE-geotiff.txt','vendor/LICENSE-proj4.txt','assets/logo-aqua.png','assets/manrope.ttf','assets/FONT-LICENSE.txt',...Array.from({length:8},(_,i)=>'model/weights-00'+i+'.bin')]);
function protect(response,isHTML=false){
 const headers=new Headers(response.headers);
 headers.set('Content-Security-Policy',CSP);
 headers.set('X-Content-Type-Options','nosniff');
 headers.set('Referrer-Policy','no-referrer');
 headers.set('Permissions-Policy','camera=(), microphone=(), geolocation=(), payment=(), usb=()');
 headers.set('Strict-Transport-Security','max-age=31536000');
 headers.set('X-Robots-Tag','noindex, nofollow');
 headers.set('Cache-Control',isHTML?'no-store':'private, max-age=900');
 return new Response(response.body,{status:response.status,statusText:response.statusText,headers});
}
export default {
 async fetch(request,env){
  if(!['GET','HEAD'].includes(request.method))return protect(new Response('Método no permitido.',{status:405,headers:{Allow:'GET, HEAD'}}));
  const url=new URL(request.url),pathname=url.pathname;
  if(pathname==='/'||pathname==='/index.html')return protect(new Response(request.method==='HEAD'?null:HTML,{headers:{'Content-Type':'text/html; charset=utf-8'}}),true);
  if(pathname.includes('%')||!ALLOWED.has(pathname.slice(1)))return protect(new Response('Recurso no encontrado.',{status:404}));
  if(!env.ASSETS?.fetch)return protect(new Response('El servicio de archivos no está disponible.',{status:503}));
  const asset=await env.ASSETS.fetch(request);
  return protect(asset,pathname==='/model/manifest.json');
 }
};
`;
fs.writeFileSync(path.join(server,'index.js'),workerSource);
console.log(JSON.stringify({worker:'dist/server/index.js',assets:'dist/client',headers:'enforced by fetch handler'}));

