# ML Portfolio - DALIS

A clean, technical portfolio website for machine learning, quantitative finance, and research-driven trading strategy development.

> A modern showcase of model engineering, signal research, and portfolio-level analysis for recruiters, hiring managers, and quantitative teams.

## Overview

This documentation-style portfolio presents four core capabilities:

- **Time series forecasting** for volatility and return behavior
- **Factor modeling** using alternative data and market signals
- **Backtesting** of strategy performance, risk, and trade economics
- **Quantitative research** combining sentiment, correlations, and statistical validation

## About Me

I am a quantitative researcher and machine learning practitioner specializing in data-driven finance. My portfolio emphasizes:

- **Technical rigor** in model design and evaluation
- **Financial intuition** for market behavior and risk
- **Production-aware architecture** for reproducibility and deployment
- **Recruiter-ready storytelling** through polished research documentation

## Featured Projects

| Project | Focus | Value Proposition |
|---|---|---|
| **Volatility Forecasting** | ML + GARCH + Monte Carlo | Predicts risk and tail outcomes with ensemble models and volatility dynamics |
| **Signal Backtesting** | Long/flat SPY strategy | Evaluates real-world strategy performance, Sharpe, drawdown, and trade statistics |
| **Correlation Research** | Google Trends + stock prices | Finds statistically significant relationships between macro/search sentiment and equities |
| **Sentiment Scanner** | Hybrid fundamental + technical + social signals | Generates actionable signals from news, Reddit, and technical indicators |

## Tech Stack

- **Languages:** Python, Markdown
- **Data:** pandas, numpy, yfinance, Polygon.io, Google Trends
- **Models:** scikit-learn, arch, Random Forest, Gradient Boosting, Logistic Regression
- **Visualization:** matplotlib, seaborn
- **Documentation:** MkDocs, Material theme, GitHub Pages

## Quick Navigation

- [Portfolio Overview](portfolio-overview.md)
- [Time Series & Forecasting](time-series-forecasting.md)
- [Factor Modeling](factor-modeling.md)
- [Backtesting](backtesting.md)
- [Quantitative Research](quantitative-research.md)
- [Machine Learning Models](machine-learning-models.md)
- [Performance Metrics](performance-metrics.md)
- [Future Improvements](future-improvements.md)

## Featured Links

- GitHub: [github.com/Jdurairaj-hub/ml-portfolio-dalis](https://github.com/Jdurairaj-hub/ml-portfolio-dalis)
- LinkedIn: [linkedin.com/in/jdurairaj](https://www.linkedin.com/in/jdurairaj/)

## Deployment

This portfolio is optimized for GitHub Pages and MkDocs.

### Local preview

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Open the local preview at `http://127.0.0.1:8000`.

### Deploy to GitHub Pages

```bash
mkdocs gh-deploy --force
```

The site publishes to:

`https://jdurairaj-hub.github.io/ml-portfolio-dalis/`

### GitHub Actions

A workflow is already configured at `.github/workflows/gh-pages.yml` to build and deploy automatically on pushes to `main`.

## Notes

- The site uses a **dark technical theme** for professional presentation.
- Navigation is optimized for recruiters and quantitative decision-makers.
- Project pages include architecture, metrics, and cross-domain research context.
