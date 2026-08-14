# AGENTS.md

## Cursor Cloud specific instructions

- This is a **zero-build, dependency-free static web app** ("Inference Ballpark", an LLM inference throughput planner). There is nothing to install and no bundler/transpiler step — the browser loads `index.html` and the plain-JS modules in `js/` directly.
- Run it in development with the command documented in `README.md`: `python3 -m http.server 8000`, then open `http://localhost:8000/`. `python3` is preinstalled on the VM. Do not open `index.html` via `file://` — the `js/*.js` modules must be served over HTTP.
- There is **no test suite, no linter, and no build command** in this repo. "Verifying" a change means loading the page and interacting with it (switch models, adjust the sliders, expand the bottom results drawer) and confirming the throughput numbers (TTFT / TPOT / tok/s / TPM) recompute.
- All logic is client-side; there is no backend/API. Editing any `js/*.js` file just requires a browser refresh (no hot-reload).
- Source layout (see `README.md` "文件" section): `js/models.js` (model dims), `js/model-graphs.js` (per-unit DAG), `js/roofline.js` (cost/throughput math), `js/walkthrough.js` (vertical DAG + play highlight), `js/app.js` (controls + rendering).
