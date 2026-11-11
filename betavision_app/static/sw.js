// BetaVision service worker: makes the app installable (Chrome / Edge "Install app").
// The page always comes from the local server when it's running; when it isn't, the
// installed app shows a short page that says how to start it. Videos and results are
// never cached here.
const CACHE = "betavision-v1";
const SHELL = ["offline.html", "style.css", "icons/icon.svg", "fonts/Manrope-latin.woff2"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.mode === "navigate") {
    e.respondWith(fetch(e.request).catch(() => caches.match("offline.html")));
  } else if (e.request.method === "GET" && url.origin === location.origin && !url.pathname.startsWith("/api/")) {
    // the offline page's own style + font (API calls, videos and frames always go straight to the server)
    e.respondWith(fetch(e.request).catch(() => caches.match(e.request, { ignoreSearch: true })
      .then((r) => r || Response.error())));
  }
});
