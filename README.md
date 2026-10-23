# 🍔 FoodDash: Online Food Delivery Analysis

A complete, single-file analytics app: **Python (pandas) cleaning → SQL database (SQLite) → interactive Streamlit dashboard**, with documentation built into the app and a hover description on every feature.

## 🚀 Live Demo

**[Open FoodDash Dashboard](https://food-delivery-analysis-obfu5gjxhjzy8witbrtjsa.streamlit.app/)**

Explore the interactive dashboard online without installing anything.

## Run it (one command)

```bash
pip install -r requirements.txt && streamlit run app.py
```

Or use the launchers, which also create a virtual environment for you:

| OS | Launcher |
|---|---|
| Windows | Double-click `run_app.bat` |
| Mac / Linux | `./run_app.sh` |

Requires Python 3.9+. The first launch cleans the data and builds the database (about 15–40 seconds). After that, the app opens instantly.

## What's inside

```text
food-delivery-analysis/
├── app.py                  # Application: cleaning, SQL layer, dashboard, docs
├── requirements.txt
├── run_app.bat / run_app.sh
├── .streamlit/config.toml  # Dark theme
├── data/
│   ├── raw/                # Source CSV dataset
│   └── processed/          # Auto-generated database and cleaned CSV
└── README.md
```

## Dashboard pages

Overview · Customers & Orders · Revenue & Profit · Delivery Performance · Restaurants & Cancellations · SQL Explorer · Data Explorer · Documentation

**Sidebar filters:** City, Cuisine, Order Status, Date Range, plus a profit-margin basis switch.

**Download Cleaned Data:** Export the full cleaned dataset as CSV.

## Tech Stack

- Python
- Pandas
- SQLite
- Streamlit
- SQL

## Features

- Automated data cleaning and preparation
- SQLite database integration
- Interactive dashboard with KPIs and visualizations
- Customer and order analysis
- Revenue and profit analysis
- Delivery performance monitoring
- Restaurant performance and cancellation analysis
- SQL Explorer and Data Explorer
- Cleaned CSV export

## Publish to GitHub

1. Create an empty repository on GitHub, leaving **Add a README** unchecked.
2. Open a terminal in the project folder.
3. Run:

```bash
git init
git add .
git commit -m "Online Food Delivery Analysis dashboard"
git branch -M main
git remote add origin https://github.com/aashika-sheik/food-delivery-analysis.git
git push -u origin main
```

If the repository is already initialized and pushed, you do not need to repeat these commands. Just commit and push your README changes.

## Deployment

Hosted using [Streamlit Community Cloud](https://share.streamlit.io/).

**Live application:** https://food-delivery-analysis-obfu5gjxhjzy8witbrtjsa.streamlit.app/

**GitHub repository:** https://github.com/aashika-sheik/food-delivery-analysis/
