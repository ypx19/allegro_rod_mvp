# Inheritable project-page fragments

These HTML pages match the public GitHub Pages site
(`docs/demo.html` → https://ypx19.github.io/allegro_rod_mvp/demo.html):
dark `#0c1218` ground, gold `#e8a54b` accent, Fraunces / Sora.

They are **not** a claim that T00 or Phase T is solved.

## Pages to inherit

1. **T00 2×2 ablation + early history-length finding**
   - File: `docs/pages/t00-ablation-history.html`
   - Assets: `20260914-t00-ablation-train-curves.png` (100k) and
     `20260914-t00-ablation-train-curves-300k.png` (copied beside the HTML)
   - 300k collapse demos: `docs/pages/t00-ablation-videos-300k/`
     (canonical: `runs/20260914-1748-t00-ablation-videos-300k/`)
   - Source plots also live under `reports/comparisons/` (do not overwrite
     the 100k PNG).

2. **T00 seed 6 contact / tilt timeline** (good demo page)
   - Canonical original (keep as-is):
     `runs/20260914-t00-seed6-tilt-collapse/index.html`
   - Inheritable copy (relative `frames/` + inlined `DATA`):
     `docs/pages/t00-seed6-tilt-collapse/index.html`
   - Sequence: 2-contact at step 29 → 1-contact at 30 → 0 at 34 →
     eval `axis_tilt` (1.2 rad) first fires at 40. Related:
     `DBG-20260823-006`, published T00 transfer replay.

3. **T00 300k A/B/C/D seed-6 analysis page**
   - Inheritable: `docs/pages/t00-ablation-demos-300k/index.html`
   - Canonical traces: `runs/20260914-1805-t00-ablation-demos-300k/`
   - Video-dir copy: `runs/20260914-1748-t00-ablation-videos-300k/analysis/`
   - Tabbed A/B/C/D **frame slider** (ckpt 306432): still + per-frame
     ω_axial / tilt / dθ/dt / n_contact / fingertip forces. MP4 is optional.
     All four 2→1→0→`axis_tilt`, no recontact. Does not overwrite page 2.

## How to publish later

1. Leave the files under `docs/pages/` so Pages serves them at
   `/pages/t00-ablation-history.html`, `/pages/t00-seed6-tilt-collapse/`,
   and `/pages/t00-ablation-demos-300k/`.
2. Add a Phase T / failure-mode section on `docs/demo.html` with cards
   linking those URLs. Do not restyle; reuse existing tokens.
3. Copy media next to the HTML. Do **not** link the public page at
   `runs/` — that tree is not published.
4. If the seed-6 original is edited, re-copy
   `runs/20260914-t00-seed6-tilt-collapse/` → `docs/pages/t00-seed6-tilt-collapse/`.
