import fs from "node:fs/promises";
import { createHash } from "node:crypto";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const base = dirname(fileURLToPath(import.meta.url));
const modules = process.argv[2] ? resolve(process.argv[2]) : join(base, "node_modules");
const vendor = join(base, "dist", "vendor");
const inventory = JSON.parse(await fs.readFile(join(vendor, "inventory.json"), "utf8"));
const sources = {
  "tf-4.22.0.min.js": "@tensorflow/tfjs/dist/tf.min.js",
  "geotiff-3.0.5.js": "geotiff/dist-browser/geotiff.js",
  "proj4-2.22.0.js": "proj4/dist/proj4.js"
};

for (const [name, relativePath] of Object.entries(sources)) {
  let bytes = await fs.readFile(join(modules, relativePath));
  if (name === "tf-4.22.0.min.js") {
    // La distribución archivada sustituyó dos llamadas dinámicas por globalThis
    // para poder mantener la política CSP. Conservamos su codificación exacta.
    const source = bytes.toString("utf8");
    bytes = Buffer.from(source
      .replace('Function("return this")()', "globalThis")
      .replace('Function("r","regeneratorRuntime = r")(t)', "globalThis.regeneratorRuntime=t")
      .replace(/\r?\n/g, "\r\n"), "utf8");
  }
  const entry = inventory.files.find(item => item.file === name);
  if (createHash("sha256").update(bytes).digest("hex") !== entry.sha256) {
    throw new Error(`La biblioteca ${name} difiere de la versión verificada.`);
  }
  await fs.writeFile(join(vendor, name), bytes);
}
console.log("Tres bibliotecas verificadas contra el inventario del detector.");
