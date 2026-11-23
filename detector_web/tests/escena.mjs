/** Prueba funcional con la ventana externa de Esmeraldas usada en octubre. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, resolve, sep } from "node:path";
import { chromium } from "../.sites-runtime/verificacion/node_modules/playwright-core/index.mjs";

const [executablePath, image, scl] = process.argv.slice(2);
if (!scl) throw new Error("Uso: node tests/escena.mjs NAVEGADOR TIFF_ESMERALDAS TIFF_SCL");
const hash = async path => createHash("sha256").update(await readFile(path)).digest("hex");
assert.equal(await hash(image), "36abe5608400a536aabfb24e21675622051acfcca2824deb6cda0547bda176b0");
assert.equal(await hash(scl), "27d0d673411d76403faa59092b939ea1236a438468256b8aabfd2e14e75bfded");

const dist = resolve("dist");
const mime = {".html":"text/html; charset=utf-8",".js":"text/javascript",".css":"text/css",".png":"image/png",".json":"application/json",".ttf":"font/ttf"};
const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  const path = resolve(dist, "." + (url.pathname === "/" ? "/index.html" : url.pathname));
  if (!path.startsWith(dist + sep) || req.method !== "GET") return res.writeHead(404).end();
  try {
    const bytes = await readFile(path);
    res.writeHead(200, {"Content-Type":mime[extname(path)] || "application/octet-stream"}).end(bytes);
  } catch { res.writeHead(404).end(); }
});
await new Promise(done => server.listen(0, "127.0.0.1", done));
let browser;
try {
  browser = await chromium.launch({executablePath, headless:true, args:["--use-angle=swiftshader","--enable-unsafe-swiftshader"]});
  const page = await browser.newPage();
  const errors = [], uploads = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => {if (request.method() !== "GET") uploads.push(request.method());});
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  await page.locator("#image").setInputFiles(image);
  await page.locator("#scl").setInputFiles(scl);
  await page.locator("#confirm-bands").check();
  await page.locator("#analyze").click();
  await page.waitForFunction(() => document.querySelector("#status").textContent.startsWith("Análisis terminado") ||
    document.querySelector("#status").classList.contains("error"), null, { timeout:240000 });
  assert.match(await page.locator("#status").innerText(), /^Análisis terminado/);
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.locator('a[download="report.json"]').click()
  ]);
  const result = JSON.parse(await readFile(await download.path(), "utf8"));
  const counts = Object.fromEntries(result.comparison.rows.map(row => [row.key, row.pixels]));
  assert.deepEqual(counts, {water:6235,background:51219,cloud:748,ponds:125});
  assert.equal(result.comparison.rows.every(row => row.metrics === null), true);
  assert.equal(result.model.threshold, 0.5);
  assert.deepEqual(errors, []);
  assert.deepEqual(uploads, []);
  const summary = { passed:true, input:"Esmeraldas 2025-02-12", model_sha256:result.model.sha256,
    browser:browser.version(), class_counts:counts, uploads:uploads.length, page_errors:errors.length,
    reference_masks_supplied:false, report_download_verified:true };
  await mkdir(".sites-runtime/checks",{recursive:true});
  await writeFile(".sites-runtime/checks/escena.json", JSON.stringify(summary,null,2)+"\n");
  console.log(summary);
} finally {
  if (browser) await browser.close();
  await new Promise(done => server.close(done));
}
