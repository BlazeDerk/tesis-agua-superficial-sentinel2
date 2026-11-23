/** Exact architecture of the frozen Keras thesis model, batch1 NHWC.
 * TensorFlow.js4.22.0 APIs: https://js.tensorflow.org/api/4.22.0/
 * No training, threshold tuning or historical Test evaluation is performed.
 */
export const EXPECTED_MODEL_SHA256 = '9e28f92b557bdd2e9a2e7b068a6622254ddbf80080926b8c004368ad5683dda4';
export const EXPECTED_SOURCE_SHA256 = 'a6e0c1a5fe44aa14fc69bc28772ad6a228f25b6d65272f2875f66cb4c83f405f';
export const EXPECTED_MANIFEST_SHA256 = '028b7f290401fd2cd2c2c7b3af6488f04f2abfac33597412f1ab875088a2f7aa';
export const PATCH_SIZE = 256;
export const CHANNEL_ORDER = Object.freeze(['B2', 'B3', 'B4', 'B8']);
const MEANS = Object.freeze([103.939, 116.779, 123.68, 114.79933333333334]);
const MAX_CHUNK_BYTES = 8 * 1024 * 1024;
const ENCODER_FILTERS = [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]];
const DECODER_FILTERS = [128, 64, 32, 16, 16];

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function getTF() {
  const tf = globalThis.tf;
  requireCondition(tf && tf.tensor4d && tf.image?.resizeBilinear,
    'Falta TensorFlow.js: carga primero vendor/tf-4.22.0.min.js.');
  requireCondition(tf.version?.tfjs === '4.22.0', 'Esta versión requiere TensorFlow.js4.22.0.');
  return tf;
}

function expectedLayers() {
  const result = [];
  let channels = 4;
  ENCODER_FILTERS.forEach((filters, block) => {
    filters.forEach((out, index) => {
      result.push({ name: `block${block + 1}_conv${index + 1}`,
        kernelShape: [3, 3, channels, out], biasShape: [out] });
      channels = out;
    });
  });
  [512, 512, 256, 128, 64].forEach((skipChannels, stage) => {
    const out = DECODER_FILTERS[stage];
    result.push({ name: `decoder${stage + 1}_conv1`,
      kernelShape: [3, 3, channels + skipChannels, out], biasShape: [out] });
    result.push({ name: `decoder${stage + 1}_conv2`,
      kernelShape: [3, 3, out, out], biasShape: [out] });
    channels = out;
  });
  result.push({ name: 'water_probability', kernelShape: [1, 1, channels, 1], biasShape: [1] });
  return result;
}

function sameShape(a, b) {
  return Array.isArray(a) && a.length === b.length && a.every((v, i) => v === b[i]);
}

function flatCount(shape) {
  return shape.reduce((total, dimension) => total * dimension, 1);
}

function relativeAssetURL(path, base) {
  requireCondition(typeof path === 'string' && /^weights-00[0-7]\.bin$/.test(path),
    'Ruta de pesos inválida.');
  const result = new URL(path, base);
  const origin = new URL(base);
  requireCondition(result.origin === origin.origin &&
    result.pathname.startsWith(origin.pathname.slice(0,origin.pathname.lastIndexOf('/')+1)) &&
    !result.search && !result.hash, 'Los pesos deben proceder del directorio del modelo.');
  return result.href;
}

async function sha256Bytes(bytes) {
  requireCondition(globalThis.crypto?.subtle,
    'La verificación de pesos requiere HTTPS o localhost con WebCrypto.');
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('');
}

async function fetchBytes(url, { signal, expectedBytes, maxBytes } = {}) {
  const response = await fetch(url, { signal, credentials: 'same-origin', redirect: 'error' });
  requireCondition(response.ok, 'No se pudo descargar el archivo ('+response.status+').');
  const declaredSize = Number(response.headers.get('content-length'));
  requireCondition(!maxBytes || !Number.isFinite(declaredSize) || declaredSize<=maxBytes,
    'El archivo supera el presupuesto de descarga.');
  const parts=[];let total=0;
  if(response.body?.getReader){
    const reader=response.body.getReader();
    try{while(true){signal?.throwIfAborted();const {done,value}=await reader.read();if(done)break;
      total+=value.byteLength;requireCondition(!maxBytes||total<=maxBytes,
        'El archivo supera el presupuesto de descarga.');parts.push(value);
    }}catch(error){await reader.cancel().catch(()=>{});throw error;}finally{reader.releaseLock();}
  }else{
    const bytes=new Uint8Array(await response.arrayBuffer());total=bytes.byteLength;
    requireCondition(!maxBytes||total<=maxBytes,'El archivo supera el presupuesto de descarga.');parts.push(bytes);
  }
  requireCondition(expectedBytes===undefined || total===expectedBytes,
    'El tamaño del archivo de pesos no coincide.');
  const bytes=new Uint8Array(total);let offset=0;for(const part of parts){bytes.set(part,offset);offset+=part.byteLength;}
  return bytes;
}

function arraySegments(record) {
  if (record.segments) return record.segments;
  return [{ chunk: record.chunk, byte_offset: record.byte_offset || 0,
    byte_length: record.byte_length, array_byte_offset: 0 }];
}

function validateManifest(manifest) {
  requireCondition(manifest && Array.isArray(manifest.arrays) && Array.isArray(manifest.chunks),
    'El manifiesto de pesos no tiene el esquema esperado.');
  requireCondition(manifest.model_sha256 === EXPECTED_MODEL_SHA256,
    'El manifiesto no corresponde al modelo final verificado de la tesis.');
  requireCondition(manifest.source_sha256 === EXPECTED_SOURCE_SHA256,
    'La procedencia del modelo no corresponde al código congelado.');
  const scale = manifest.input_scale ?? manifest.architecture?.input_scale;
  const means = manifest.means ?? manifest.MEANS_BGR_NIR ?? manifest.architecture?.means_bgr_nir ?? manifest.architecture?.means;
  requireCondition(scale === 10000, 'Normalización del modelo incompatible.');
  requireCondition(sameShape(means, MEANS), 'Medias BGR/NIR incompatibles.');
  const inputShape = manifest.input_shape ?? manifest.architecture?.input_shape;
  const outputShape = manifest.output_shape ?? manifest.architecture?.output_shape;
  requireCondition(sameShape(inputShape, [null, 256, 256, 4]) || sameShape(inputShape, [256, 256, 4]),
    'Dimensiones de entrada incompatibles.');
  requireCondition(sameShape(outputShape, [null, 256, 256, 1]) || sameShape(outputShape, [256, 256, 1]),
    'Dimensiones de salida incompatibles.');
  requireCondition(manifest.arrays.length === 48, 'Se requieren48 matrices de kernel/bias.');

  requireCondition(manifest.chunks.length===8,'El modelo requiere ocho fragmentos de pesos.');
  const chunkMap = new Map();
  manifest.chunks.forEach(record => {
    const name = record.file ?? record.name;
    requireCondition(typeof name === 'string' && /^weights-00[0-7]\.bin$/.test(name) && !chunkMap.has(name), 'Fragmento de pesos duplicado.');
    requireCondition(Number.isSafeInteger(record.byte_length) && record.byte_length > 0 &&
      record.byte_length <= MAX_CHUNK_BYTES && record.byte_length % 4 === 0,
      'Tamaño de fragmento inválido.');
    requireCondition(/^[a-f0-9]{64}$/.test(record.sha256), 'Falta SHA256 del fragmento.');
    chunkMap.set(name, record);
  });
  const arrays = new Map();
  manifest.arrays.forEach(record => {
    const key = `${record.layer}/${record.name}`;
    requireCondition(!arrays.has(key) && record.dtype === 'float32', 'Matriz duplicada o tipo incompatible.');
    requireCondition(Array.isArray(record.shape) && record.shape.every(n => Number.isSafeInteger(n) && n > 0),
      'Shape de matriz inválido.');
    const bytes = flatCount(record.shape) * 4;
    requireCondition(record.byte_length === bytes && /^[a-f0-9]{64}$/.test(record.sha256),
      'Tamaño o SHA256 de matriz inválido.');
    const segments = arraySegments(record).slice().sort((a, b) => a.array_byte_offset - b.array_byte_offset);
    let previousEnd = 0;
    segments.forEach(segment => {
      const chunk = chunkMap.get(segment.chunk);
      requireCondition(chunk && Number.isSafeInteger(segment.byte_offset) && segment.byte_offset >= 0 &&
        Number.isSafeInteger(segment.byte_length) && segment.byte_length > 0 &&
        segment.byte_offset % 4 === 0 && segment.byte_length % 4 === 0 &&
        segment.byte_offset + segment.byte_length <= chunk.byte_length &&
        segment.array_byte_offset === previousEnd,
        'Segmentación de una matriz de pesos inválida.');
      previousEnd += segment.byte_length;
    });
    requireCondition(previousEnd === bytes, 'Matriz de pesos incompleta.');
    arrays.set(key, { record, segments, bytes });
  });
  expectedLayers().forEach(layer => {
    const kernel = arrays.get(`${layer.name}/kernel`);
    const bias = arrays.get(`${layer.name}/bias`);
    requireCondition(kernel && bias && sameShape(kernel.record.shape, layer.kernelShape) &&
      sameShape(bias.record.shape, layer.biasShape), `Pesos incompatibles en ${layer.name}.`);
  });
  return { arrays, chunkMap };
}

async function selectBackend(tf, backend) {
  if (backend === 'auto') {
    try {
      tf.env().set('WEBGL_FORCE_F16_TEXTURES', false);
      const enabled = await tf.setBackend('webgl');
      if (!enabled) throw new Error('WebGL no disponible.');
      await tf.ready();
      requireCondition(tf.env().getBool('WEBGL_RENDER_FLOAT32_CAPABLE'), 'WebGLfloat32 no disponible.');
      return;
    } catch {
      await tf.setBackend('cpu');
      await tf.ready();
      return;
    }
  }
  if (backend === 'webgl') tf.env().set('WEBGL_FORCE_F16_TEXTURES', false);
  requireCondition(await tf.setBackend(backend), `El backend ${backend} no está disponible.`);
  await tf.ready();
  if (backend === 'webgl') {
    requireCondition(tf.env().getBool('WEBGL_RENDER_FLOAT32_CAPABLE'),
      'Se requiere WebGL con texturas float32; prueba backend CPU.');
  }
}

/** Download/authenticate frozen float32 weights, with ≤8MiB requests.
 * options: backend ('auto'|'webgl'|'cpu'), signal, onProgress(record).
 * Call returned model.dispose() when it is no longer needed.
 */
export async function loadModel(manifestURL, options = {}) {
  const tf = getTF();
  const url = new URL(manifestURL, globalThis.location?.href || 'http://localhost/').href;
  requireCondition(new URL(url).origin===new URL(globalThis.location?.href||url).origin,
    'El manifiesto debe proceder del mismo sitio.');
  const manifestBytes=await fetchBytes(url,{signal:options.signal,maxBytes:256*1024});
  requireCondition(await sha256Bytes(manifestBytes)===EXPECTED_MANIFEST_SHA256,
    'El manifiesto del modelo no coincide con la versión verificada.');
  const manifest = JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(manifestBytes));
  const { arrays, chunkMap } = validateManifest(manifest);
  await selectBackend(tf, options.backend || 'auto');
  const storage = new Map();
  const tensors = new Map();
  const progress = options.onProgress || (() => {});
  try {
    arrays.forEach((value, name) => storage.set(name, new Uint8Array(value.bytes)));
    let downloaded = 0;
    const total = Array.from(chunkMap.values()).reduce((sum, chunk) => sum + chunk.byte_length, 0);
    for (const [name, record] of chunkMap) {
      const bytes = await fetchBytes(relativeAssetURL(name, url), {
        signal: options.signal, expectedBytes: record.byte_length, maxBytes: MAX_CHUNK_BYTES
      });
      requireCondition(await sha256Bytes(bytes) === record.sha256, `SHA256 incorrecto en ${name}.`);
      arrays.forEach((value, arrayName) => {
        value.segments.filter(segment => segment.chunk === name).forEach(segment => {
          storage.get(arrayName).set(bytes.subarray(segment.byte_offset,
            segment.byte_offset + segment.byte_length), segment.array_byte_offset);
        });
      });
      downloaded += bytes.byteLength;
      progress({ stage: 'weights', loaded: downloaded, total, chunk: name });
    }
    // Browser/Node on supported x86/ARM platforms use little-endian float32.
    const endianProbe = new Uint8Array(new Uint32Array([0x01020304]).buffer);
    requireCondition(endianProbe[0] === 4, 'Plataforma big-endian no compatible.');
    for (const [name, { record }] of arrays) {
      const bytes = storage.get(name);
      requireCondition(await sha256Bytes(bytes) === record.sha256, `SHA256 incorrecto en ${name}.`);
      const values = new Float32Array(bytes.buffer);
      requireCondition(values.every(Number.isFinite), `Pesos no finitos en ${name}.`);
      tensors.set(name, tf.tensor(values, record.shape, 'float32'));
      storage.delete(name);
    }
    let disposed = false;
    let busy = false;
    const model = {
      manifest,
      model_sha256: manifest.model_sha256,
      source_sha256: manifest.source_sha256,
      backend: tf.getBackend(),
      input_shape: [1, 256, 256, 4],
      output_shape: [1, 256, 256, 1],
      dispose() {
        requireCondition(!busy, 'Espera a que termine la predicción antes de liberar el modelo.');
        if (!disposed) {
          tensors.forEach(tensor => tensor.dispose());
          tensors.clear();
          disposed = true;
        }
      },
      async predictDN(dn, predictionOptions = {}) {
        requireCondition(!disposed, 'El modelo ya fue liberado.');
        requireCondition(!busy, 'Procesa un parche cada vez; ya hay una predicción activa.');
        requireCondition(tf.getBackend() === model.backend, 'No cambies el backend después de cargar el modelo.');
        busy = true;
        try {
          return await runForward(tf, tensors, dn, predictionOptions, model);
        } finally {
          busy = false;
        }
      }
    };
    progress({ stage: 'ready', loaded: total, total, backend: model.backend, model_sha256: model.model_sha256 });
    return Object.freeze(model);
  } catch (error) {
    tensors.forEach(tensor => tensor.dispose());
    storage.clear();
    throw error;
  }
}

async function runForward(tf, weights, dn, options, model) {
  requireCondition(ArrayBuffer.isView(dn) && !(dn instanceof DataView) &&
    dn.length === PATCH_SIZE * PATCH_SIZE * 4,
    'La predicción requiere256×256×4 valores DN intercalados B2/B3/B4/B8.');
  requireCondition(dn.every(value => Number.isFinite(value) && value >= 0 && value < 65535),
    'Los DN deben ser finitos y no incluir NoData65535; excluye/rellena esos píxeles antes de predecir.');
  if (options.signal?.aborted) throw new DOMException('Predicción cancelada.', 'AbortError');
  const start = performance.now();
  const alive = new Set();
  const skips = [];
  const keep = tensor => { alive.add(tensor); return tensor; };
  const release = tensor => { if (tensor && alive.delete(tensor)) tensor.dispose(); };
  const progress = options.onProgress || (() => {});
  const conv = (input, name, sigmoid = false) => keep(tf.tidy(() => {
    const result = tf.add(tf.conv2d(input, weights.get(`${name}/kernel`), 1, 'same', 'NHWC'),
      weights.get(`${name}/bias`));
    return sigmoid ? tf.sigmoid(result) : tf.relu(result);
  }));
  let x;
  try {
    x = keep(tf.tidy(() => tf.sub(
      tf.mul(tf.div(tf.tensor4d(Float32Array.from(dn), [1, 256, 256, 4], 'float32'), 10000), 255),
      tf.tensor1d(MEANS, 'float32'))));
    for (let block = 0; block < ENCODER_FILTERS.length; block++) {
      for (let index = 0; index < ENCODER_FILTERS[block].length; index++) {
        const previous = x;
        x = conv(x, `block${block + 1}_conv${index + 1}`);
        release(previous);
      }
      skips.push(x);
      x = keep(tf.tidy(() => tf.maxPool(x, [2, 2], [2, 2], 'valid')));
      progress({ stage: 'encoder', block: block + 1, total: 5 });
      if (options.signal?.aborted) throw new DOMException('Predicción cancelada.', 'AbortError');
      if (options.yieldBetweenStages !== false) await tf.nextFrame();
    }
    for (let stage = 0; stage < 5; stage++) {
      const skip = skips.pop();
      const previous = x;
      x = keep(tf.tidy(() => {
        // Keras3 UpSampling2D bilinear uses halfPixelCenters=true.
        const up = tf.image.resizeBilinear(previous, [skip.shape[1], skip.shape[2]], false, true);
        return tf.concat([up, skip], 3);
      }));
      release(previous);
      release(skip);
      for (let index = 1; index <= 2; index++) {
        const previousConv = x;
        x = conv(x, `decoder${stage + 1}_conv${index}`);
        release(previousConv);
      }
      progress({ stage: 'decoder', block: stage + 1, total: 5 });
      if (options.signal?.aborted) throw new DOMException('Predicción cancelada.', 'AbortError');
      if (options.yieldBetweenStages !== false) await tf.nextFrame();
    }
    const last = x;
    x = conv(x, 'water_probability', true);
    release(last);
    const data = Float32Array.from(await x.data());
    requireCondition(data.length === 65536 && data.every(value => Number.isFinite(value) && value >= 0 && value <= 1),
      'La predicción produjo valores o dimensiones inválidas.');
    return { data, width: 256, height: 256, channels: 1,
      model_sha256: model.model_sha256, backend: model.backend, elapsed_ms: performance.now() - start };
  } finally {
    alive.forEach(tensor => tensor.dispose());
    alive.clear();
  }
}

/** Return {data:Float32Array,width:256,height:256,model_sha256,backend,elapsed_ms}. */
export async function predictDN(model, dn, options = {}) {
  requireCondition(model && typeof model.predictDN === 'function', 'Carga primero el modelo.');
  return await model.predictDN(dn, options);
}
