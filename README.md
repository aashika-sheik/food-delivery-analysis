# 🍔 FoodDash: Online Food Delivery Analysis

A complete, single-file analytics app: **Python (pandas) cleaning → SQL database (SQLite) → interactive Streamlit dashboard**,
with documentation built into the app and a hover description on every feature.

## Run it (one command)

```bash
pip install -r requirements.txt && streamlit run app.py
```

Or use the launchers, which also create a virtual environment for you:

| OS | Launcher |
|---|---|
| Windows | double-click `run_app.bat` |
| Mac / Linux | `./run_app.sh` |

Requires Python 3.9+. The first launch cleans the data and builds the database (about 15-40 seconds). After that the app opens instantly.

## What's inside

```
food-delivery-analysis/
├── app.py                  # the whole application (cleaning, SQL layer, dashboard, docs)
├── requirements.txt
├── run_app.bat / run_app.sh
├── .streamlit/config.toml  # dark theme
├── data/
│   ├── raw/                # your CSV (replace it to analyse another file)
│   └── processed/          # auto-generated: food_delivery.db + cleaned_orders.csv
└── README.md
```

## Dashboard pages

Overview · Customers & Orders · Revenue & Profit · Delivery Performance · Restaurants & Cancellations · SQL Explorer · Data Explorer · Documentation

Sidebar filters: City, Cuisine, Order status, Date range, plus a profit-margin basis switch. **Download Cleaned Data** exports the full cleaned dataset as CSV.

## Publish to GitHub

1. Create an empty repo on github.com (**New repository**, leave "Add a README" unchecked).
2. In this folder:

```bash
git init
git add .
git commit -m "Online Food Delivery Analysis dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/food-delivery-analysis.git
git push -u origin main
```

Optional free hosting: https://share.streamlit.io → New app → pick the repo → main file `app.py`.
