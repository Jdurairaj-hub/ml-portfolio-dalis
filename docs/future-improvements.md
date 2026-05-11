# Future Improvements

This portfolio is designed to evolve into a production-grade research platform. Future enhancements include both research and deployment upgrades.

## Research roadmap

- **Rolling factor analysis** to measure evolving signal strength
- **Lagged predictors** to test lead/lag relationships
- **Alternative data integration** such as option flow or macro indicators
- **Automated feature selection** for model robustness
- **Hyperparameter search** for optimal model tuning

## Productization goals

- **API service** for live signal requests
- **Deployment pipeline** for continuous model updates
- **Dashboard visualization** with interactive charts
- **Expanded coverage** across asset classes and sectors

## Technical improvements

- Add `mkdocs-material` theming and improve site visuals
- Standardize documentation with a site-wide navigation structure
- Build GitHub Actions automation for deployment
- Add test-driven validation for model pipelines

## Deployment guide

The documentation can be deployed using:

```bash
pip install mkdocs mkdocs-material
mkdocs gh-deploy --force
```

The GitHub Actions workflow at `.github/workflows/gh-pages.yml` builds and deploys when code is pushed to `main`.

## Recruiter-friendly polish

- Maintain concise, actionable bullet points
- Use tables and example output for readability
- Emphasize business value and technical contribution
- Keep the site modern, minimal, and research-focused

> This page is the roadmap for the next evolution of the ML + finance portfolio website.
