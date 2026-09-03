// Stark's service worker: what makes the app installable, and what makes it
// start (and keep painting) with no network.
//
// Two strategies, because Stark's files fall into exactly two kinds:
//
//   * **The shell** — the navigation response. It names the hashed wasm/JS of
//     one particular build, so a stale copy pins the whole app to that build.
//     Network-first: the network answers when it can, the cache when it can't.
//   * **Everything else same-origin** — the wasm, the JS glue, the stylesheet,
//     the brush stamps, the substrate height maps, the HDR environment. Every one
//     of those is content-addressed by the asset pipeline (or by Stark itself:
//     a stamp *is* the hash of its coverage, §6.6), so its URL changes when its
//     bytes do and a hit can never be stale. Cache-first, revalidating in the
//     background so a renamed file is in hand before it is asked for.
//
// The large runtime fetches are what this is really for. `builtins::import_all`,
// `substrates::open_default` and the environment HDR are all fetched *after* the
// wasm boots and are deliberately not in the binary — without a cache they are
// a fresh megabyte-scale download every start, and offline they simply fail.
//
// Bump `VERSION` when this file's logic changes; `activate` sweeps every cache
// that does not carry the current prefix.

"use strict";

const VERSION = "v1";
const CACHE = `stark::${VERSION}`;

// The shell, plus the pieces a launcher wants before the app has run once.
// Not the wasm or the JS glue: those carry a content hash in their names, which
// only the generated index.html knows, so they arrive through `fetch` below.
const SHELL = ["./", "./manifest.json", "./icon-192.png", "./icon-512.png"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) =>
      // `reload`, so an install triggered by a deploy cannot be served the very
      // HTTP-cached shell it is meant to replace.
      cache.addAll(SHELL.map((url) => new Request(url, { cache: "reload" })))
    )
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
      )
      // Take over the page that installed us, so the first visit is already
      // offline-capable rather than the second.
      .then(() => self.clients.claim())
  );
});

/** Whether a response is ours to keep. Opaque and partial responses are not. */
function cacheable(response) {
  return response && response.status === 200 && response.type === "basic";
}

self.addEventListener("fetch", (event) => {
  const request = event.request;

  // Only GET is idempotent enough to replay from a cache. A range request is
  // answered 206, which the Cache API refuses to store — let it go to the
  // network untouched rather than fail it.
  if (request.method !== "GET" || request.headers.has("range")) {
    return;
  }

  const url = new URL(request.url);
  // Same-origin only. Cross-origin GETs come back opaque, so caching them would
  // store a response we cannot inspect; and the collaboration transport (§12.4)
  // is a WebSocket, which never reaches here at all.
  if (url.origin !== self.location.origin) {
    return;
  }

  // The shell: fresh when we can be, cached when we cannot.
  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request)
        .then((response) => {
          if (cacheable(response)) {
            const copy = response.clone();
            caches.open(CACHE).then((cache) => cache.put(request, copy));
          }
          return response;
        })
        .catch(() =>
          caches
            .match(request)
            .then((cached) => cached || caches.match("./"))
            .then(
              (cached) =>
                cached ||
                new Response("<h1>Stark is offline</h1>", {
                  status: 503,
                  statusText: "Service Unavailable",
                  headers: { "Content-Type": "text/html; charset=utf-8" },
                })
            )
        )
    );
    return;
  }

  // Everything else: the cached copy now, a fresher one for next time.
  event.respondWith(
    caches.match(request).then((cached) => {
      const networked = fetch(request)
        .then((response) => {
          if (cacheable(response)) {
            const copy = response.clone();
            caches.open(CACHE).then((cache) => cache.put(request, copy));
          }
          return response;
        })
        // Offline with nothing cached is a genuine failure for a subresource —
        // the caller (wasm fetch, <script>, image decode) has its own handling,
        // and a synthesized 503 body would only confuse it.
        .catch(() => cached || Response.error());
      return cached || networked;
    })
  );
});
