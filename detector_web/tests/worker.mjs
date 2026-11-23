import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
const worker=(await import(pathToFileURL(process.cwd()+'/dist/server/index.js'))).default;
const names=['Content-Security-Policy','X-Content-Type-Options','Referrer-Policy','Permissions-Policy','Strict-Transport-Security','X-Robots-Tag'];
let checks=0;
for(const [url,method,code] of [['https://site.test/','GET',200],['https://site.test/','HEAD',200],['https://site.test/','POST',405],['https://site.test/secret','GET',404],['https://site.test/%2e%2e/private','GET',404]]){
 const r=await worker.fetch(new Request(url,{method}),{});assert.equal(r.status,code);for(const n of names)assert.ok(r.headers.get(n));checks++;
}
let r=await worker.fetch(new Request('https://site.test/app.js?v=2'),{ASSETS:{fetch:async()=>new Response('asset',{headers:{'Content-Type':'application/javascript'}})}});
assert.equal(await r.text(),'asset');assert.equal(r.headers.get('X-Content-Type-Options'),'nosniff');checks++;
r=await worker.fetch(new Request('https://site.test/'),{});assert.ok((await r.text()).includes('Carga tu escena'));assert.equal(r.headers.get('Cache-Control'),'no-store');checks++;
assert.equal((await worker.fetch(new Request('https://site.test/model/manifest.json'),{})).status,503);checks++;
console.log(JSON.stringify({workerChecks:checks,passed:true}));
await fs.writeFile('.sites-runtime/checks/worker.json',JSON.stringify({checks,passed:true}));

