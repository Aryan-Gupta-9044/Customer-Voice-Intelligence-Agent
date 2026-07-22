# Deploying This App

## Option A: GitHub Pages (fully static, in-browser)

GitHub Pages only serves static files — it can't run a Python server. This repo
uses [**stlite**](https://github.com/whitphx/stlite), which ports Streamlit to
WebAssembly so it runs entirely inside the visitor's browser via Pyodide. No
server, no cost beyond GitHub Pages itself.

**This works well for this specific app** because every agent already has a
pure-Python fallback (hashing-based embeddings, a lexicon-based sentiment
model, an in-memory vector store) for when the heavier ML packages
(`torch`, `transformers`, `sentence-transformers`, `chromadb`) aren't
available — and those heavy packages can't be installed in a browser anyway
(Pyodide only supports pure-Python wheels). The `docs/` folder is already set
up to run this way.

### Steps

1. Push this whole folder to a GitHub repository.
2. In the repo settings: **Settings → Pages → Build and deployment → Source:
   "Deploy from a branch"**, branch `main` (or whichever you use), folder
   `/docs`. Save.
3. GitHub will publish `https://<your-username>.github.io/<repo-name>/`
   within a minute or two. Open it — the app boots directly in the browser
   (first load takes ~10-20s while Pyodide + packages download; it's cached
   after that).
4. **If you edit `app.py` or anything under `agents/`, `database/`, or
   `utils/`**, run `./sync_docs.sh` before committing — `docs/` is a self
   contained copy of the source (GitHub Pages can only serve files inside the
   published folder, so the runtime files are mirrored there rather than
   referenced from outside `docs/`).

### Limitations of this path

- **No `GEMINI_API_KEY`.** Everything runs client-side and is visible in the
  browser, so never put a secret key in `docs/` or `index.html`. The Q&A tab
  will use the deterministic fallback answerer, not the Gemini-backed one.
- **No real ChromaDB / sentence-transformers / transformers / torch.** These
  have compiled (non pure-Python) parts that Pyodide can't install. The app
  already falls back gracefully to lightweight equivalents — see the "Why the
  Q&A Answers Were Inaccurate" section above for how those fallbacks work.
- **Large CSVs will be slow.** Pyodide runs Python single-threaded in the
  browser, so tens of thousands of rows will feel sluggish. For a GitHub
  Pages demo, sample your dataset down (see `--sample` in
  `utils/prepare_kaggle_dataset.py`) to a few hundred–low thousands of rows.
- The bundled `st.column_config.ProgressColumn` / newer widgets depend on the
  Streamlit version stlite currently bundles — if something in the Themes
  table doesn't render, it's usually safe to drop that one `column_config`
  call and fall back to a plain `st.dataframe`.

## Option B: Streamlit Community Cloud (full server, still free)

Not literally "GitHub Pages," but the practical alternative when you want the
full experience — real `chromadb`, `sentence-transformers`, a `GEMINI_API_KEY`
secret, and no dataset-size caveats:

1. Push the repo to GitHub (same repo works — Community Cloud reads `app.py`
   and `requirements.txt` from the root, and ignores `docs/`).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click "New app", pick the repo/branch and `app.py` as the entry
   point.
3. In the app's **Settings → Secrets**, add:
   ```
   GEMINI_API_KEY = "your-key-here"
   ```
4. Deploy. Community Cloud installs everything in `requirements.txt`
   (including the optional heavy packages), so you get real embeddings and
   LLM-backed answers with no code changes.
