# Session Compaction Summary

## User Intent
- Build a PWA cat-detection app ("Un Gaaat") that uses a local ONNX model to identify cats in photos
- Score cats by colour rarity and build a persistent collection (localStorage), exportable as JSON
- Polish UI: Inter + Phosphor icons, no emojis, pixelated cat thumbnails, good iOS Safari behaviour

## Contextual Work Summary

### Architecture
- Single-page PWA with three screens: loading, camera, list — no router, just class toggling
- No service worker (removed early; was causing Safari cache pain during development)
- All ML inference local via `@huggingface/transformers@3` (CDN ES module), model `Xenova/yolos-tiny`

### ML & Scoring
- WebGPU attempted on non-iOS desktop (Chrome), WASM quantized int8 on iOS (forced `numThreads=1`)
- iOS detection: UA check + `ontouchstart in window` (catches iPad in desktop-mode which fakes Mac UA)
- Images resized to ≤640px before detection to avoid OOM on mobile
- Score formula: `min(10, rarityBase + (confidence − 0.65) × 3.5)` — rarity sets floor (4–9.5), confidence adds ±1.2 variation; avoids always-10 saturation since YOLOS confidence is typically >0.9
- Rarity table: Ginger/Black = 4.0–4.5 (Common), Silver/White = 5.5 (Uncommon), Golden/Slate Blue = 7.5–8.5 (Rare), Lilac = 9.5 (Very Rare)

### Thumbnail Storage
- Bounding box extracted from original full-res canvas, scaled back from detection-size coordinates
- Thumbnail: aspect-ratio-preserving, max 32px on longest side, stored as PNG data URL in localStorage entry
- Displayed with `object-fit: contain`, `image-rendering: pixelated / crisp-edges`, dominant colour as background fill

### Collection & Data
- localStorage key `un-gaaat-cats`, array of entries: `{id, date, colorName, colorHex, textColor, confidence, rarityBase, rarityLabel, catometer, thumbnail}`
- Export: JSON download; Import: JSON file, merges by id (no duplicates), newer entries first
- Delete per-card via Phosphor `ph-x` button

### UI / UX Flow
- Always opens on list screen; empty state = onboarding splash with large icon, app description ("quantized YOLOS-tiny ONNX model running locally"), arrow hint to shutter
- List header (with import/export) hidden when collection is empty
- Camera opened only when user taps shutter; landscape blocker only shown on touch devices (`body.is-touch`)
- Scanning overlay shows blurred frozen frame + random phrase from `SCAN_PHRASES` (10 entries incl. "pst pst pst", "Hiding this from ALF…")
- No cat found → toast message, return to camera

### Fonts & Icons
- Inter `@font-face` defined directly in `app.css`, referencing `../tessella/fonts/InterDisplay-*.woff2`
- Phosphor Light linked via `<link>` in HTML (`fonts/phosphor/phosphor.css`)
- No `@import`, no copied font files, no emoji in UI
- App icon (`media/icon.png`, `media/icon-512.png`, `media/favicon.ico`) used in: loading screen (bouncing), list header (28px), empty state (140px); backgrounds behind icon are `#000` to match icon's black background

### Card Design
- 2-column grid; each card: `border-radius: 16px`, pixelated thumbnail on top (`aspect-ratio: 4/3`, `object-fit: contain`), colour info panel below
- Panel: colour name, cat-o-meter bar, rarity badge, date

## Files Touched

### Core
- **app.js**: All logic — model loading, camera, detection, scoring, collection CRUD, rendering, export/import, scan phrases
- **index.html**: Three screens + scanning overlay + landscape blocker; Phosphor link; media icons wired
- **app.css**: Full styles; Inter @font-face; no @import; landscape blocker gated on `body.is-touch`

### Assets
- **manifest.json**: Icons wired to `media/icon.png` and `media/icon-512.png`
- **media/**: `icon.png` (192px), `icon-512.png`, `favicon.ico` — added by user
- **fonts/phosphor/**: `phosphor.css` + `Phosphor-Light.woff2` — added by user

### Removed
- **sw.js**: Service worker deleted (was causing Safari hard-refresh pain during dev)
- **icons/icon.svg**: Auto-generated SVG removed at user request
