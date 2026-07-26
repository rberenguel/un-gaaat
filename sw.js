const VERSION = 'v1';
const CACHE   = `un-gaaat-${VERSION}`;

const SHELL = [
  './',
  './index.html',
  './manifest.json',
  './src/app.js',
  './src/app.css',
  './fonts/phosphor/phosphor.css',
  './fonts/phosphor/Phosphor-Light.woff2',
  './media/icon.png',
  './media/icon-512.png',
  './media/favicon.ico',
];

const MODEL_FILES = [
  './models/Xenova/yolos-tiny/config.json',
  './models/Xenova/yolos-tiny/preprocessor_config.json',
  './models/Xenova/yolos-tiny/onnx/model_q4.onnx',
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(async cache => {
      // Cache shell eagerly; model files may not exist yet during first install
      await cache.addAll(SHELL);
      await Promise.allSettled(MODEL_FILES.map(f => cache.add(f)));
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
  );
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  const url = new URL(e.request.url);
  if (url.origin !== self.location.origin) return;
  e.respondWith(
    caches.match(e.request).then(cached => cached || fetch(e.request))
  );
});
