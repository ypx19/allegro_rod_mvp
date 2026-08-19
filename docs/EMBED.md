# Embedding the research demo

Standalone page: [`demo.html`](demo.html) (also served as [`index.html`](index.html) for GitHub Pages `/docs`).

Videos live under [`media/`](media/) with relative paths — works on GitHub Pages, local `python -m http.server`, and static hosts.

## Local preview

```bash
cd docs
python -m http.server 8765
# open http://127.0.0.1:8765/demo.html
```

## GitHub Pages

Repo **Settings → Pages → Deploy from branch**, folder `/docs`.

URL shape: `https://<user>.github.io/<repo>/` → loads `docs/index.html` → redirects to `demo.html`.

## Personal website (iframe)

Use embed mode to hide the sticky nav and full-viewport hero chrome:

```html
<iframe
  src="https://ypx19.github.io/allegro_rod_mvp/demo.html?embed=1"
  title="Rod Rotation MVP research demo"
  loading="lazy"
  style="width:100%;min-height:85vh;border:0;border-radius:12px;background:#0c1218;"
></iframe>
```

Optional: pass your repo URL so the hero shows a Repository button:

```text
demo.html?embed=1&repo=https://github.com/ypx19/allegro_rod_mvp
```

## Personal website (section copy)

Prefer owning the layout on your site? Link out instead of iframeing:

```html
<section>
  <h2>Rod Rotation MVP</h2>
  <p>Three-finger axial twisting under tip constraints — successes, cliffs, and ablations.</p>
  <a href="https://ypx19.github.io/allegro_rod_mvp/demo.html">Open interactive demo</a>
</section>
```

CSS variables on the page are scoped under plain selectors on `body`; for shadow-free reuse, stick to the iframe or a full-page link (recommended).

## README-style video embeds (GitHub)

GitHub Markdown can play checked-in MP4s:

```markdown
### Success — top tip @s=400

https://github.com/YOU/allegro_rod_mvp/assets/... 
<!-- or relative after push: -->

https://raw.githubusercontent.com/YOU/allegro_rod_mvp/main/docs/media/success-top-tip-c4-twisting.mp4
```

Relative paths in README may not play on all GitHub UIs; the HTML demo is the reliable viewer.
