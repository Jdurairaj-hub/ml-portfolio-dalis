# Deployment

This site is configured for GitHub Pages deployment using MkDocs Material.

## Local preview

Install dependencies and run the local server:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser.

## GitHub Pages deployment

Use MkDocs to publish the site directly to GitHub Pages:

```bash
mkdocs gh-deploy --force
```

This command will build the site and push the generated contents to the `gh-pages` branch.

## GitHub Actions

A GitHub Actions workflow is included at:

- `.github/workflows/gh-pages.yml`

The workflow automatically:

- installs Python and site dependencies
- builds the MkDocs site
- deploys to GitHub Pages on `main` branch pushes

## Site URL

The site is published at:

`https://jdurairaj-hub.github.io/ml-portfolio-dalis/`

## Notes

- `mkdocs build` validates the configuration locally
- `mkdocs serve` enables live editing and preview
- The deployment workflow is optimized for GitHub Pages
