# ML Portfolio - DALIS

A comprehensive machine learning portfolio showcasing financial modeling, quantitative analysis, and algorithmic trading strategies. Built for DALIS, this repository demonstrates expertise in data-driven finance through multiple interconnected projects.

## Team

This portfolio was developed collaboratively by:

**Portfolio Managers:**
- John Victor ([john.victor@dal.ca](mailto:john.victor@dal.ca))- Portfolio Manager
- Ibaad Hemani ([ipadhemani@gmail.com](mailto:ipadhemani@gmail.com)) - Portfolio Manager

**Analysts:**
- Sarah Finkle ([sarahfinkle@dal.ca](mailto:sarahfinkle@dal.ca)) - Senior Analyst
- Apaar ([apaar.nagi@dal.ca](mailto:apaar.nagi@dal.ca)) - Junior Analyst
- Marko Dimitrijevic ([pn564459@dal.ca](mailto:pn564459@dal.ca)) - Junior Analyst

*Special thanks to the entire team for their contributions to the ML models, data analysis, and project architecture!*

## �📊 Projects Overview

### 1. Asset Correlation Analysis
**Location:** `asset-correlation-analysis/`

Analyzes correlations between oil prices and transportation sector stocks, plus stock price movements vs. Google Trends data.

- **oilScreener.py**: Fetches oil (USO) and transportation stock data from Polygon.io, calculates correlations and performs linear regression analysis.
- **screener_sarah.py**: Correlates stock prices with Google Trends data for keywords like "AI" and "tariffs".

**Key Features:**
- Real-time data fetching from Polygon.io API
- Correlation matrices and visualization
- Google Trends integration

### 2. Prediction Model
**Location:** `prediction-model/`

Machine learning-based trading strategy for SPY (S&P 500 ETF) using a simple long/flat approach.

- **ML_Simple_LongFlat_SPY.ipynb**: Jupyter notebook implementing logistic regression model with technical indicators (RSI, moving averages) to predict market direction.

**Key Features:**
- Feature engineering with technical indicators
- Train/test split validation
- Strategy backtesting framework

### 3. Sentiment Analysis
**Location:** `sentiment-analysis/`

Financial sentiment analysis combining news articles and social media data from Reddit.

- **main.py**: Comprehensive sentiment analyzer that scrapes financial news and Reddit posts, applies positive/negative word scoring, and generates sentiment metrics.

**Key Features:**
- Multi-source data collection (news APIs, Reddit API)
- Custom sentiment scoring algorithm
- Time-weighted sentiment analysis
- Recency filtering (configurable age limits)

### 4. Volatility Forecasting
**Location:** `volatility-forecasting/`

Advanced volatility prediction using machine learning and GARCH models with Monte Carlo simulation.

- **main.py**: Orchestrates the entire pipeline
- **src/data.py**: Polygon.io data fetching
- **src/features.py**: Technical indicator creation and risk metrics
- **src/model.py**: Ensemble ML model training (Random Forest + Gradient Boosting + Ridge)
- **src/garch.py**: GARCH volatility modeling
- **src/simulation.py**: Three Monte Carlo simulation methods

**Key Features:**
- ML-enhanced volatility forecasting
- GARCH model integration
- Monte Carlo price simulation
- Risk metrics calculation
- Comprehensive visualization

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Polygon.io API key (free tier available)
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-portfolio-dalis
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Copy .env.example to .env and fill in your API keys
cp .env.example .env
# Edit .env with your Polygon.io API key
```

### Running Projects

Each project can be run independently:

```bash
# Volatility Forecasting
cd volatility-forecasting
python main.py

# Sentiment Analysis
cd ../sentiment-analysis
python main.py --help  # See available options

# Asset Correlation (run individual scripts)
cd ../asset-correlation-analysis
python oilScreener.py
python screener_sarah.py

# Prediction Model (open in Jupyter)
cd ../prediction-model
jupyter notebook ML_Simple_LongFlat_SPY.ipynb
```

## 📈 Key Technologies

- **Data Sources:** Polygon.io, Yahoo Finance, Google Trends, Reddit API, News APIs
- **ML Frameworks:** scikit-learn, arch (GARCH), joblib
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **APIs:** requests, yfinance, pytrends

## 🏗️ Architecture Principles

- **Modular Design:** Each project is self-contained with clear separation of concerns
- **Environment Management:** Centralized requirements and environment variables
- **Error Handling:** Robust API rate limiting and error recovery
- **Reproducibility:** Configurable random seeds and parameter overrides
- **Documentation:** Comprehensive READMEs and inline code documentation

## 📝 Development Standards

This portfolio maintains high code quality standards:

- **Code Style:** PEP 8 compliant Python
- **Version Control:** Git with meaningful commit messages
- **Documentation:** Inline docstrings and project READMEs
- **Testing:** Modular code structure enabling unit testing
- **Security:** API keys stored in environment variables, .gitignore configured

## 🤝 Contributing

This is a portfolio project. For improvements or questions, please open an issue or submit a pull request.


## �📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 DALIS ML Portfolio Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

*Built with ❤️ for demonstrating ML expertise in quantitative finance*
