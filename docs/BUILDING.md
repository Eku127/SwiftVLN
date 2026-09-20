# Build the documentation

The SwiftVLN Wiki follows SatNav's documentation setup: Sphinx, MyST Markdown,
and the Read the Docs theme. The source pages live in `docs/en-US/` and
`docs/zh-CN/`, and share images under `docs/assets/`.

## Build and preview locally

Use Python 3.12 and run these commands from the repository root. The
documentation environment only needs the packages in `docs/requirements.txt`.

Linux and macOS:

```bash
python3 -m venv .local/docs-venv
.local/docs-venv/bin/python -m pip install -r docs/requirements.txt
.local/docs-venv/bin/python scripts/build_docs.py
.local/docs-venv/bin/python -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html
```

Windows PowerShell:

```powershell
py -3.12 -m venv .local/docs-venv
.local/docs-venv/Scripts/python.exe -m pip install -r docs/requirements.txt
.local/docs-venv/Scripts/python.exe scripts/build_docs.py
.local/docs-venv/Scripts/python.exe -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html
```

Open <http://localhost:8000/wiki/>. The English and Chinese homepages are
`/wiki/en-US/index.html` and `/wiki/zh-CN/index.html`. The sidebar language link
opens the corresponding page in the other language. Each language has its own
search index, navigation, and copy buttons for code blocks.

## Update the Wiki

Implementation diagrams live in `docs/assets/concepts/diagrams/`; architecture
and workflow diagrams live in `docs/assets/workflows/`. Edit the
native `.drawio` sources and export the matching light-theme `.svg` files in
both languages. The [asset notes](assets/concepts/README.md) list the diagrams
and an export command; [workflow asset notes](assets/workflows/README.md) map
the workflow figures to their code references. The Wiki displays SVGs and links the editable sources.

Use `python scripts/export_diagrams.py --drawio /path/to/drawio` to export all
diagrams as compact SVGs with native text. On Windows, pass the path to
`draw.io.exe`. Keep `html=0;whiteSpace=nowrap;` in label styles and set line
breaks explicitly. Editable sources remain in the separate `.drawio` files.

1. Edit the existing Markdown in the relevant language directory.
2. For a new page, add its counterpart in the other language and register both
   pages in their language's `index.md` toctree.
3. Keep shared figures in `docs/assets/` and use relative links in Markdown.
4. Run `scripts/build_docs.py` and preview the result locally.
5. Commit the source changes and push to `master`.

The build script prepares temporary source copies in `docs/_build/sources/`
and builds both languages with Sphinx warnings treated as errors. It adapts
source-code links to GitHub, keeps language switches within the Wiki, and
copies shared assets into the generated site. The original Markdown remains
readable on GitHub. Generated files and `.local/` environments are ignored by Git.

## Publish on GitHub Pages

The **Deploy documentation** workflow in `.github/workflows/docs.yml` builds
and publishes `docs/_build/html/` when `docs/**`, `scripts/build_docs.py`, or
the workflow changes on `master`. GitHub Actions uploads the generated site
as a Pages artifact and deploys it directly.

The public entry points are:

- [SwiftVLN Wiki](https://eku127.github.io/SwiftVLN/wiki/)
- [English](https://eku127.github.io/SwiftVLN/wiki/en-US/index.html)
- [简体中文](https://eku127.github.io/SwiftVLN/wiki/zh-CN/index.html)

After pushing, open [Actions → Deploy documentation](https://github.com/Eku127/SwiftVLN/actions/workflows/docs.yml)
to check the build and deployment. To publish the current `master` again,
select **Run workflow** on that page.

The repository's **Settings → Pages → Build and deployment → Source** is
configured as **GitHub Actions**. Publishing the Wiki uses the repository's
existing visibility setting; the Pages site is publicly accessible, matching
SatNav's setup.

The site root redirects to `wiki/`, and language-page URLs directly under the
root redirect to the matching Wiki page. All site assets use relative paths,
so the output also works at a domain root or under a project subpath.
