"""
FoodDash - Online Food Delivery Analysis (single-file Streamlit app)
====================================================================
Run:   streamlit run app.py

What this file does
-------------------
1. Loads the raw CSV from ./data/raw/ (or lets you upload one).
2. Cleans it (missing values, outliers, invalid values, standardisation, logic checks).
3. Engineers features (day type, peak hour, profit margin %, age group, ...).
4. Stores the result in a normalised SQLite database (./data/processed/food_delivery.db).
   This is the app's SQL layer: every chart/KPI is a SQL query against the view `vw_orders`.
5. Shows an interactive dashboard with filters, KPIs, tabs/pages, a SQL explorer,
   a data explorer, and full documentation.

The database is built once and re-used on every later start, so the app opens instantly.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import sqlite3
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------------
# 0. CONFIGURATION
# ----------------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
RAW_DIR = APP_DIR / "data" / "raw"
PROC_DIR = APP_DIR / "data" / "processed"
DB_PATH = PROC_DIR / "food_delivery.db"
CLEAN_CSV = PROC_DIR / "cleaned_orders.csv"
PIPELINE_VERSION = "1.0"

# Business assumptions used for the engineered "delivery delay" feature.
# expected delivery time = BASE_PREP_MIN + MIN_PER_KM * distance_km
BASE_PREP_MIN = 20.0
MIN_PER_KM = 2.5

# Colour palette (matches the FoodDash mock-up)
ORANGE, RED, BLUE, GREEN, PURPLE, GREY, YELLOW, PINK = (
    "#FF8A3D", "#FF5C7A", "#4C8DFF", "#2ED8A3", "#9B6BFF", "#8892A6", "#FFC93C", "#FF6FB5")
PALETTE = [ORANGE, RED, BLUE, GREEN, PURPLE, GREY, YELLOW, PINK]

CITY_COORDS = {  # lat, lon
    "Delhi": (28.61, 77.21), "Mumbai": (19.08, 72.88), "Bangalore": (12.97, 77.59),
    "Hyderabad": (17.39, 78.49), "Chennai": (13.08, 80.27),
}
AGE_ORDER = ["Under 18", "18-25", "26-35", "36-45", "46-60", "60+", "Unknown"]
DISTANCE_ORDER = ["<=5 km", "5-10 km", "10-20 km", "20-30 km", ">30 km"]
TIME_BAND_ORDER = ["<=45 min", "46-90 min", "91-150 min", "151-210 min", ">210 min"]
PERF_ORDER = ["Fast", "Standard", "Slow", "Very Slow"]
DISC_PCT_ORDER = ["No discount", "0-5%", "5-10%", "10-20%", ">20%"]


# ----------------------------------------------------------------------------------
# 1. DATA CLEANING & FEATURE ENGINEERING
# ----------------------------------------------------------------------------------
PLACEHOLDERS = {"", "nan", "none", "null", "n/a", "na", "-", "--", "?"}
CITY_MAP = {"bengaluru": "Bangalore", "bangalore": "Bangalore", "bombay": "Mumbai",
            "mumbai": "Mumbai", "new delhi": "Delhi", "delhi": "Delhi", "madras": "Chennai",
            "chennai": "Chennai", "hyderabad": "Hyderabad"}
PAYMENT_MAP = {"upi": "UPI", "cod": "COD", "cash on delivery": "COD", "cash": "COD",
               "card": "Card", "credit card": "Card", "debit card": "Card", "wallet": "Wallet"}
STATUS_MAP = {"delivered": "Delivered", "cancelled": "Cancelled", "canceled": "Cancelled"}
GENDER_MAP = {"male": "Male", "m": "Male", "female": "Female", "f": "Female", "other": "Other"}
REASON_MAP = {"late delivery": "Late Delivery", "customer cancelled": "Customer Cancelled",
              "restaurant issue": "Restaurant Issue"}


def _std_text(series: pd.Series, mapping: dict | None = None) -> pd.Series:
    """Trim, collapse spaces, turn placeholders into NaN and standardise the casing."""
    s = series.astype(object).where(series.notna(), np.nan)
    s = s.str.strip().str.replace(r"\s+", " ", regex=True)
    low = s.str.lower()
    s = s.mask(low.isin(PLACEHOLDERS))
    out = s.str.title()
    if mapping:
        mapped = low.map(mapping)
        out = mapped.where(mapped.notna(), out)
    return out.where(s.notna(), np.nan)


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Return (clean_df, cleaning_log, quality_report)."""
    log: list[dict] = []

    def add(step: str, rule: str, n) -> None:
        log.append({"Step": step, "Rule applied": rule, "Values affected": int(n)})

    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    missing_before = df.isna().sum()
    n_raw = len(df)

    # --- 1. duplicates -------------------------------------------------------------
    df = df.drop_duplicates(subset="order_id", keep="first").reset_index(drop=True)
    add("1. Duplicates", "Dropped duplicate order_id rows (kept first)", n_raw - len(df))

    # --- 2. standardise categoricals -----------------------------------------------
    changed = 0
    for col, mp in [("city", CITY_MAP), ("area", None), ("cuisine_type", None),
                    ("payment_mode", PAYMENT_MAP), ("customer_gender", GENDER_MAP),
                    ("order_status", STATUS_MAP), ("cancellation_reason", REASON_MAP),
                    ("order_day", None)]:
        new = _std_text(df[col], mp)
        changed += int(((new != df[col]) & df[col].notna() & new.notna()).sum()
                       + (df[col].notna() & new.isna()).sum())
        df[col] = new
    for col in ["order_id", "customer_id", "restaurant_id", "delivery_partner_id"]:
        df[col] = df[col].astype(str).str.strip().str.upper()
    df["restaurant_name"] = df["restaurant_name"].astype(str).str.strip()
    add("2. Standardise categories",
        "Trim/collapse spaces, unify spelling & casing (city, cuisine, payment, status, ...)", changed)

    # --- 3. numeric coercion --------------------------------------------------------
    num_cols = ["customer_age", "delivery_time_min", "distance_km", "order_value",
                "discount_applied", "final_amount", "delivery_rating", "restaurant_rating",
                "profit_margin"]
    bad_numeric = 0
    for c in num_cols:
        before = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        bad_numeric += int(df[c].isna().sum() - before)
    add("3. Numeric parsing", "Non-numeric text in numeric columns converted to missing", bad_numeric)

    # --- 4. dates / times -----------------------------------------------------------
    dt = pd.to_datetime(df["order_date"], format="%m/%d/%Y", errors="coerce")
    unparsed = dt.isna() & df["order_date"].notna()
    if unparsed.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt.loc[unparsed] = pd.to_datetime(df.loc[unparsed, "order_date"], errors="coerce")
    add("4. Dates", "Parsed order_date (MM/DD/YYYY, with fallback parser). The count shown is dates that could NOT "
        "be parsed; missing dates stay missing and are never guessed", int(dt.isna().sum() - df["order_date"].isna().sum()))
    tm = pd.to_datetime(df["order_time"], format="%H:%M", errors="coerce")
    hour = tm.dt.hour
    if hour.dropna().nunique() <= 1:
        df["order_hour"] = np.nan
        add("4b. Time", "order_time holds only 00:00 in this file, so no hour-of-day can be derived; "
            "the supplied Peak_Hour flag is used instead", int(df["order_time"].notna().sum()))
    else:
        df["order_hour"] = hour

    # --- 5. invalid values ----------------------------------------------------------
    inv = 0
    m = ~df["customer_age"].between(10, 100) & df["customer_age"].notna()
    inv += int(m.sum()); df.loc[m, "customer_age"] = np.nan
    for c, lo in [("distance_km", 0), ("delivery_time_min", 0), ("order_value", 0)]:
        m = (df[c] <= lo) & df[c].notna()
        inv += int(m.sum()); df.loc[m, c] = np.nan
    m = (df["discount_applied"] < 0) & df["discount_applied"].notna()
    inv += int(m.sum()); df.loc[m, "discount_applied"] = np.nan
    add("5a. Invalid values", "Impossible age (<10 or >100), non-positive distance/time/order value, "
        "negative discount -> missing", inv)
    over5 = int((df["restaurant_rating"] > 5).sum() + (df["delivery_rating"] > 5).sum())
    df["restaurant_rating"] = df["restaurant_rating"].clip(1, 5)
    df["delivery_rating"] = df["delivery_rating"].clip(1, 5)
    add("5b. Ratings", "Ratings above 5 capped at 5 (restaurant_rating had values up to 5.5)", over5)

    # --- 6. logical consistency -----------------------------------------------------
    cancelled = df["order_status"].eq("Cancelled")
    n_rated_cancel = int((cancelled & df["delivery_rating"].notna()).sum())
    df.loc[cancelled, "delivery_rating"] = np.nan
    add("6a. Cancelled vs rating", "Cancelled orders cannot have a delivery rating -> rating removed", n_rated_cancel)
    n_reason_delivered = int((~cancelled & df["cancellation_reason"].notna()).sum())
    df.loc[~cancelled, "cancellation_reason"] = np.nan
    n_reason_missing = int((cancelled & df["cancellation_reason"].isna()).sum())
    df.loc[cancelled & df["cancellation_reason"].isna(), "cancellation_reason"] = "Not Specified"
    add("6b. Cancellation reason", "Delivered orders keep no reason; cancelled orders with no reason -> 'Not Specified'",
        n_reason_delivered + n_reason_missing)
    derived_day = np.where(dt.dt.dayofweek >= 5, "Weekend", "Weekday")
    df["order_day_type"] = pd.Series(derived_day, index=df.index).where(dt.notna(), df["order_day"])
    df["order_day_type"] = df["order_day_type"].fillna("Unknown")
    add("6c. Day type", "Weekday/Weekend recomputed from the order date (original flag used when the date is missing)",
        int((dt.notna() & (df["order_day"] != df["order_day_type"])).sum()))

    # --- 7. amounts: reconstruct what can be derived exactly --------------------------
    ov, di, fa = df["order_value"], df["discount_applied"], df["final_amount"]
    m1 = ov.isna() & di.notna() & fa.notna()
    ov = ov.mask(m1, fa + di)
    m2 = di.isna() & ov.notna() & fa.notna()
    di = di.mask(m2, (ov - fa).clip(lower=0))
    add("7a. Reconstruct amounts", "order_value = final + discount, discount = order_value - final "
        "(exact arithmetic, no estimation)", int(m1.sum() + m2.sum()))

    # --- 8. outlier capping (IQR fences, 1.5 x IQR) -----------------------------------
    capped = 0
    for c, series in [("delivery_time_min", df["delivery_time_min"]), ("distance_km", df["distance_km"]),
                      ("order_value", ov)]:
        q1, q3 = series.quantile(.25), series.quantile(.75)
        lo, hi = max(q1 - 1.5 * (q3 - q1), 0), q3 + 1.5 * (q3 - q1)
        n_out = int(((series < lo) | (series > hi)).sum())
        capped += n_out
        clipped = series.clip(lo, hi)
        if c == "order_value":
            ov = clipped
        else:
            df[c] = clipped
    add("8. Outliers", "Delivery time, distance and order value winsorised at the 1.5 x IQR fences", capped)

    # --- 9. missing-value imputation --------------------------------------------------
    df["age_imputed"] = df["customer_age"].isna().astype(int)
    df["delivery_time_imputed"] = df["delivery_time_min"].isna().astype(int)
    df["distance_imputed"] = df["distance_km"].isna().astype(int)
    df["order_value_imputed"] = (ov.isna() & fa.isna()).astype(int)
    df["age_group"] = pd.cut(df["customer_age"], bins=[0, 17, 25, 35, 45, 60, 130],
                             labels=AGE_ORDER[:6]).astype(object)
    df["age_group"] = df["age_group"].where(df["customer_age"].notna(), "Unknown")
    # Age group is built BEFORE imputing age, so 'Unknown' stays honest instead of a fake median bucket.
    df["customer_age"] = df["customer_age"].fillna(df["customer_age"].median()).round(0)
    df["delivery_time_min"] = df["delivery_time_min"].fillna(df["delivery_time_min"].median())
    df["distance_km"] = df["distance_km"].fillna(df["distance_km"].median())
    di = di.fillna(di.median())
    mfa = ov.isna() & fa.notna()
    ov = ov.mask(mfa, fa + di)
    ov = ov.fillna(ov.median())
    add("9a. Imputation (numeric)", "Median fill: customer_age, delivery_time_min, distance_km, discount, order_value "
        "(imputed rows are flagged with *_imputed columns)",
        int(df[["age_imputed", "delivery_time_imputed", "distance_imputed", "order_value_imputed"]].sum().sum()))
    cat_fill = 0
    for c in ["city", "area", "cuisine_type", "payment_mode", "customer_gender"]:
        cat_fill += int(df[c].isna().sum())
        df[c] = df[c].fillna("Unknown")
    add("9b. Imputation (categorical)", "Missing city/area/cuisine/payment/gender -> 'Unknown' (mode-fill would invent "
        "~17% of a category)", cat_fill)
    delivered = df["order_status"].eq("Delivered")
    n_rating_fill = int((delivered & df["delivery_rating"].isna()).sum())
    df.loc[delivered, "delivery_rating"] = df.loc[delivered, "delivery_rating"].fillna(
        df.loc[delivered, "delivery_rating"].median())
    add("9c. Imputation (ratings)", "Delivered orders with no rating -> median rating; cancelled stay empty", n_rating_fill)
    peak_raw = df["peak_hour"].astype(str).str.strip().str.lower().map(
        {"true": 1, "1": 1, "yes": 1, "y": 1, "false": 0, "0": 0, "no": 0, "n": 0})
    df["is_peak_hour"] = peak_raw
    df["peak_hour_flag"] = peak_raw.map({1: "Peak", 0: "Off-Peak"}).fillna("Unknown")
    add("9d. Peak hour", "Peak_Hour kept as supplied; missing stays 'Unknown' (no clock time exists to recompute it)",
        int(peak_raw.isna().sum()))

    # --- 10. final amounts, discount sanity -------------------------------------------
    n_disc_over = int((di > ov).sum())
    di = np.minimum(di, ov)
    fa_new = (ov - di).clip(lower=0)
    df["order_value"], df["discount_applied"], df["final_amount"] = ov.round(2), di.round(2), fa_new.round(2)
    add("10. Amount logic", "Discount cannot exceed order value (capped); final_amount = order_value - discount "
        "(removes negative final amounts)", n_disc_over)

    # --- 11. profit margin ------------------------------------------------------------
    pm = df["profit_margin"].fillna(df["profit_margin"].median())
    if pm.abs().max() > 1.5:
        pm = pm / 100.0
    df["profit_margin_raw_pct"] = (pm * 100).round(2)
    n_neg = int((pm < 0).sum())
    df["profit_margin_pct"] = df["profit_margin_raw_pct"].clip(lower=0)
    df["profit_amount"] = (df["final_amount"] * df["profit_margin_pct"] / 100).round(2)
    df["profit_amount_raw"] = (df["final_amount"] * df["profit_margin_raw_pct"] / 100).round(2)
    add("11. Profit margin", "Negative margins floored at 0% in 'profit_margin_pct' (raw value kept in "
        "'profit_margin_raw_pct'; the sidebar lets you switch basis)", n_neg)

    # --- 12. feature engineering ------------------------------------------------------
    df["order_date"] = dt.dt.strftime("%Y-%m-%d")
    df["order_month"] = dt.dt.strftime("%Y-%m")
    df["is_cancelled"] = (df["order_status"] == "Cancelled").astype(int)
    df["discount_pct"] = (df["discount_applied"] / df["order_value"] * 100).clip(0, 100).round(2)
    df["discount_band"] = pd.cut(df["discount_pct"], bins=[-1, 0, 5, 10, 20, 1000],
                                 labels=DISC_PCT_ORDER).astype(object)
    df["distance_band"] = pd.cut(df["distance_km"], bins=[0, 5, 10, 20, 30, 1e9],
                                 labels=DISTANCE_ORDER).astype(object)
    df["delivery_time_band"] = pd.cut(df["delivery_time_min"], bins=[0, 45, 90, 150, 210, 1e9],
                                      labels=TIME_BAND_ORDER).astype(object)
    df["delivery_performance"] = pd.cut(df["delivery_time_min"], bins=[0, 45, 90, 150, 1e9],
                                        labels=PERF_ORDER).astype(object)
    df["expected_delivery_min"] = (BASE_PREP_MIN + MIN_PER_KM * df["distance_km"]).round(1)
    df["delivery_delay_min"] = (df["delivery_time_min"] - df["expected_delivery_min"]).round(1)
    df["is_late"] = (df["delivery_delay_min"] > 0).astype(int)
    add("12. Feature engineering",
        "Added order_day_type, peak_hour_flag, profit_margin_pct, profit_amount, age_group, discount_pct/band, "
        "distance/time bands, delivery_performance, expected_delivery_min, delivery_delay_min, is_late, order_month", 0)

    cols = ["order_id", "customer_id", "customer_age", "age_group", "customer_gender", "city", "area",
            "restaurant_id", "restaurant_name", "cuisine_type", "delivery_partner_id", "payment_mode",
            "order_date", "order_month", "order_day_type", "peak_hour_flag", "is_peak_hour",
            "delivery_time_min", "distance_km", "distance_band", "delivery_time_band", "delivery_performance",
            "expected_delivery_min", "delivery_delay_min", "is_late", "order_value", "discount_applied",
            "discount_pct", "discount_band", "final_amount", "order_status", "is_cancelled",
            "cancellation_reason", "delivery_rating", "restaurant_rating", "profit_margin_raw_pct",
            "profit_margin_pct", "profit_amount", "profit_amount_raw", "age_imputed",
            "delivery_time_imputed", "distance_imputed", "order_value_imputed"]
    out = df[cols].copy()

    quality = []
    alias = {"order_day": "order_day_type", "profit_margin": "profit_margin_pct"}
    for c in raw.columns:
        k = c.strip().lower()
        tgt = alias.get(k, k)
        if k == "peak_hour":
            after = int((out["peak_hour_flag"] == "Unknown").sum())
        elif tgt in out.columns:
            after = int(out[tgt].isna().sum())
        else:
            after = None                      # column is not carried into the clean table (order_time)
        quality.append({"Column": c, "Missing before": int(missing_before[k]), "Missing after": after,
                        "Missing % before": round(missing_before[k] / n_raw * 100, 1)})
    return out, log, quality


# ----------------------------------------------------------------------------------
# 2. SQL LAYER  (normalised SQLite schema + analytical view)
# ----------------------------------------------------------------------------------
SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
CREATE TABLE dim_city          (city_id INTEGER PRIMARY KEY, city_name TEXT NOT NULL UNIQUE);
CREATE TABLE dim_area          (area_id INTEGER PRIMARY KEY, area_name TEXT NOT NULL UNIQUE);
CREATE TABLE dim_cuisine       (cuisine_id INTEGER PRIMARY KEY, cuisine_name TEXT NOT NULL UNIQUE);
CREATE TABLE dim_payment_mode  (payment_id INTEGER PRIMARY KEY, payment_mode TEXT NOT NULL UNIQUE);
CREATE TABLE dim_customer      (customer_id TEXT PRIMARY KEY, first_order_date TEXT);
CREATE TABLE dim_restaurant    (restaurant_id TEXT PRIMARY KEY, restaurant_name TEXT);
CREATE TABLE dim_delivery_partner (partner_id TEXT PRIMARY KEY);

CREATE TABLE orders (
    order_id               TEXT PRIMARY KEY,
    customer_id            TEXT    NOT NULL REFERENCES dim_customer(customer_id),
    restaurant_id          TEXT    NOT NULL REFERENCES dim_restaurant(restaurant_id),
    partner_id             TEXT    NOT NULL REFERENCES dim_delivery_partner(partner_id),
    city_id                INTEGER NOT NULL REFERENCES dim_city(city_id),
    area_id                INTEGER NOT NULL REFERENCES dim_area(area_id),
    cuisine_id             INTEGER NOT NULL REFERENCES dim_cuisine(cuisine_id),
    payment_id             INTEGER NOT NULL REFERENCES dim_payment_mode(payment_id),
    customer_age           REAL,
    age_group              TEXT,
    customer_gender        TEXT,
    order_date             TEXT,
    order_month            TEXT,
    order_day_type         TEXT CHECK (order_day_type IN ('Weekday','Weekend','Unknown')),
    peak_hour_flag         TEXT,
    is_peak_hour           INTEGER,
    delivery_time_min      REAL,
    distance_km            REAL,
    distance_band          TEXT,
    delivery_time_band     TEXT,
    delivery_performance   TEXT,
    expected_delivery_min  REAL,
    delivery_delay_min     REAL,
    is_late                INTEGER,
    order_value            REAL,
    discount_applied       REAL,
    discount_pct           REAL,
    discount_band          TEXT,
    final_amount           REAL,
    order_status           TEXT NOT NULL CHECK (order_status IN ('Delivered','Cancelled')),
    is_cancelled           INTEGER NOT NULL,
    cancellation_reason    TEXT,
    delivery_rating        REAL CHECK (delivery_rating IS NULL OR delivery_rating BETWEEN 1 AND 5),
    restaurant_rating      REAL CHECK (restaurant_rating BETWEEN 1 AND 5),
    profit_margin_raw_pct  REAL,
    profit_margin_pct      REAL,
    profit_amount          REAL,
    profit_amount_raw      REAL,
    age_imputed            INTEGER,
    delivery_time_imputed  INTEGER,
    distance_imputed       INTEGER,
    order_value_imputed    INTEGER
);
CREATE INDEX ix_orders_month   ON orders(order_month);
CREATE INDEX ix_orders_city    ON orders(city_id);
CREATE INDEX ix_orders_cuisine ON orders(cuisine_id);
CREATE INDEX ix_orders_cust    ON orders(customer_id);
CREATE INDEX ix_orders_rest    ON orders(restaurant_id);
CREATE INDEX ix_orders_status  ON orders(order_status);

CREATE VIEW vw_orders AS
SELECT o.order_id, o.customer_id, o.customer_age, o.age_group, o.customer_gender,
       c.city_name AS city, a.area_name AS area,
       o.restaurant_id, r.restaurant_name, cu.cuisine_name AS cuisine_type,
       o.partner_id AS delivery_partner_id, p.payment_mode,
       o.order_date, o.order_month, o.order_day_type, o.peak_hour_flag, o.is_peak_hour,
       o.delivery_time_min, o.distance_km, o.distance_band, o.delivery_time_band, o.delivery_performance,
       o.expected_delivery_min, o.delivery_delay_min, o.is_late,
       o.order_value, o.discount_applied, o.discount_pct, o.discount_band, o.final_amount,
       o.order_status, o.is_cancelled, o.cancellation_reason, o.delivery_rating, o.restaurant_rating,
       o.profit_margin_raw_pct, o.profit_margin_pct, o.profit_amount, o.profit_amount_raw,
       o.age_imputed, o.delivery_time_imputed, o.distance_imputed, o.order_value_imputed
FROM orders o
JOIN dim_city c          ON c.city_id = o.city_id
JOIN dim_area a          ON a.area_id = o.area_id
JOIN dim_cuisine cu      ON cu.cuisine_id = o.cuisine_id
JOIN dim_payment_mode p  ON p.payment_id = o.payment_id
JOIN dim_restaurant r    ON r.restaurant_id = o.restaurant_id;

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""

ERD_DOT = """
digraph G { rankdir=LR; bgcolor="transparent";
 node [shape=record, style="filled", fillcolor="#18213f", color="#4C8DFF", fontcolor="white", fontname="Helvetica"];
 edge [color="#8892A6"];
 orders [label="{orders (fact)|order_id PK\\lcustomer_id FK\\lrestaurant_id FK\\lpartner_id FK\\lcity_id FK\\larea_id FK\\lcuisine_id FK\\lpayment_id FK\\l...metrics, ratings, flags\\l}", fillcolor="#2a1f4d", color="#9B6BFF"];
 dim_customer [label="{dim_customer|customer_id PK\\lfirst_order_date\\l}"];
 dim_restaurant [label="{dim_restaurant|restaurant_id PK\\lrestaurant_name\\l}"];
 dim_delivery_partner [label="{dim_delivery_partner|partner_id PK\\l}"];
 dim_city [label="{dim_city|city_id PK\\lcity_name\\l}"];
 dim_area [label="{dim_area|area_id PK\\larea_name\\l}"];
 dim_cuisine [label="{dim_cuisine|cuisine_id PK\\lcuisine_name\\l}"];
 dim_payment_mode [label="{dim_payment_mode|payment_id PK\\lpayment_mode\\l}"];
 orders -> dim_customer; orders -> dim_restaurant; orders -> dim_delivery_partner;
 orders -> dim_city; orders -> dim_area; orders -> dim_cuisine; orders -> dim_payment_mode;
}
"""


def _dim(df: pd.DataFrame, col: str, id_col: str, name_col: str):
    vals = sorted(df[col].dropna().unique())
    d = pd.DataFrame({id_col: range(1, len(vals) + 1), name_col: vals})
    return d, df[col].map(dict(zip(d[name_col], d[id_col])))


def build_database(clean: pd.DataFrame, log: list, quality: list, sig: str, n_raw: int) -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DB_PATH.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    try:
        con.executescript(SCHEMA_SQL)
        d_city, city_id = _dim(clean, "city", "city_id", "city_name")
        d_area, area_id = _dim(clean, "area", "area_id", "area_name")
        d_cui, cuisine_id = _dim(clean, "cuisine_type", "cuisine_id", "cuisine_name")
        d_pay, pay_id = _dim(clean, "payment_mode", "payment_id", "payment_mode")
        first = clean.groupby("customer_id")["order_date"].min().reset_index()
        first.columns = ["customer_id", "first_order_date"]
        rest = (clean.groupby("restaurant_id")["restaurant_name"]
                .agg(lambda s: s.mode().iat[0]).reset_index())
        partners = pd.DataFrame({"partner_id": sorted(clean["delivery_partner_id"].unique())})
        for name, d in [("dim_city", d_city), ("dim_area", d_area), ("dim_cuisine", d_cui),
                        ("dim_payment_mode", d_pay), ("dim_customer", first),
                        ("dim_restaurant", rest), ("dim_delivery_partner", partners)]:
            d.to_sql(name, con, if_exists="append", index=False)
        o = clean.copy()
        o["city_id"], o["area_id"], o["cuisine_id"], o["payment_id"] = city_id, area_id, cuisine_id, pay_id
        o["partner_id"] = o["delivery_partner_id"]
        order_cols = [r[1] for r in con.execute("PRAGMA table_info(orders)")]
        o[order_cols].to_sql("orders", con, if_exists="append", index=False, chunksize=20000)
        bad_fk = con.execute("PRAGMA foreign_key_check").fetchall()
        assert not bad_fk, f"Foreign-key violations: {bad_fk[:3]}"
        meta = {"signature": sig, "built_at": datetime.now().strftime("%b %d, %Y  %I:%M %p"),
                "raw_rows": str(n_raw), "clean_rows": str(len(clean)),
                "cleaning_log": json.dumps(log), "quality": json.dumps(quality)}
        con.executemany("INSERT INTO meta VALUES (?,?)", list(meta.items()))
        con.commit()
    finally:
        con.close()
    os.replace(tmp, DB_PATH)
    flat = sql_df("SELECT * FROM vw_orders")
    flat.to_csv(CLEAN_CSV, index=False)


def sql_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run a read-only query against the app database."""
    con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()


def find_raw_csv() -> Path | None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def file_signature(path: Path) -> str:
    s = path.stat()
    return hashlib.md5(f"{path.name}|{s.st_size}|{int(s.st_mtime)}|{PIPELINE_VERSION}".encode()).hexdigest()[:16]


@st.cache_resource(show_spinner="First run: cleaning data and building the SQL database (one-time, ~15-40 s)...")
def prepare_database(raw_path: str, sig: str) -> dict:
    """Build the DB if it does not exist (or the CSV changed). Otherwise reuse it instantly."""
    if DB_PATH.exists() and CLEAN_CSV.exists():
        try:
            con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
            stored = dict(con.execute("SELECT key, value FROM meta").fetchall())
            con.close()
            if stored.get("signature") == sig:
                return {"sig": sig, "rebuilt": False}
        except sqlite3.Error:
            pass
    raw = pd.read_csv(raw_path, low_memory=False)
    clean, log, quality = clean_data(raw)
    build_database(clean, log, quality, sig, len(raw))
    return {"sig": sig, "rebuilt": True}


CTX = {"sig": ""}


@st.cache_data(show_spinner=False, max_entries=600)
def cached_query(sig: str, sql: str, params: tuple) -> pd.DataFrame:
    return sql_df(sql, params)


@st.cache_data(show_spinner=False)
def get_meta(sig: str) -> dict:
    return dict(sql_df("SELECT key, value FROM meta").values.tolist())


@st.cache_data(show_spinner=False)
def cleaned_csv_bytes(sig: str) -> bytes:
    return CLEAN_CSV.read_bytes()


# ---- filter -> WHERE clause ----------------------------------------------------------
def where_clause(f: dict, extra: str | None = None) -> tuple[str, tuple]:
    conds, params = [], []
    for col, key in (("city", "cities"), ("cuisine_type", "cuisines"), ("order_status", "statuses")):
        vals = f.get(key)
        if vals is not None:
            conds.append(f"{col} IN ({','.join('?' * len(vals))})")
            params += list(vals)
    if f.get("date_from"):
        conds.append("order_date BETWEEN ? AND ?")
        params += [f["date_from"], f["date_to"]]
    if extra:
        conds.append(extra)
    return ("WHERE " + " AND ".join(conds)) if conds else "", tuple(params)


def run(sql: str, f: dict, extra: str | None = None) -> pd.DataFrame:
    """Execute an analytical SQL template. `/*W*/` is replaced by the sidebar filters."""
    w, params = where_clause(f, extra)
    sql = sql.replace("/*W*/", w).replace("@PROFIT@", f["profit_col"])
    return cached_query(CTX["sig"], sql, params)


DELIVERED = "order_status = 'Delivered'"
KPI_COLS = """COUNT(*) AS orders,
 SUM(CASE WHEN order_status='Delivered' THEN final_amount END) AS revenue,
 AVG(CASE WHEN order_status='Delivered' THEN final_amount END) AS aov,
 AVG(CASE WHEN order_status='Delivered' THEN delivery_time_min END) AS avg_delivery_time,
 AVG(is_cancelled)*100.0 AS cancel_rate,
 AVG(delivery_rating) AS avg_rating,
 SUM(CASE WHEN order_status='Delivered' THEN @PROFIT@ END)*100.0
   / NULLIF(SUM(CASE WHEN order_status='Delivered' THEN final_amount END),0) AS margin"""

SQL = {
    "kpi": f"SELECT {KPI_COLS}, COUNT(DISTINCT customer_id) AS customers FROM vw_orders /*W*/",
    "monthly": f"SELECT order_month, {KPI_COLS} FROM vw_orders /*W*/ GROUP BY order_month ORDER BY order_month",
    "by_cuisine": f"SELECT cuisine_type, {KPI_COLS} FROM vw_orders /*W*/ GROUP BY cuisine_type ORDER BY orders DESC",
    "by_city": f"SELECT city, {KPI_COLS}, AVG(CASE WHEN order_status='Delivered' THEN is_late END)*100.0 AS late_pct "
               f"FROM vw_orders /*W*/ GROUP BY city ORDER BY orders DESC",
    "top_customers": """SELECT customer_id, COUNT(*) AS orders,
        SUM(CASE WHEN order_status='Delivered' THEN final_amount END) AS revenue,
        AVG(CASE WHEN order_status='Delivered' THEN final_amount END) AS aov,
        AVG(is_cancelled)*100.0 AS cancel_rate
        FROM vw_orders /*W*/ GROUP BY customer_id ORDER BY revenue DESC LIMIT ?""",
    "by_age": f"SELECT age_group, {KPI_COLS} FROM vw_orders /*W*/ GROUP BY age_group",
    "by_gender": f"SELECT customer_gender, {KPI_COLS} FROM vw_orders /*W*/ GROUP BY customer_gender",
    "by_daytype": f"SELECT order_day_type, COUNT(DISTINCT order_date) AS days, {KPI_COLS} "
                  f"FROM vw_orders /*W*/ GROUP BY order_day_type",
    "orders_per_customer": """SELECT n AS orders_per_customer, COUNT(*) AS customers FROM
        (SELECT customer_id, COUNT(*) AS n FROM vw_orders /*W*/ GROUP BY customer_id) GROUP BY n ORDER BY n""",
    "payment": f"SELECT payment_mode, {KPI_COLS} FROM vw_orders /*W*/ GROUP BY payment_mode ORDER BY orders DESC",
    "cancel_reasons": "SELECT cancellation_reason, COUNT(*) AS cancelled FROM vw_orders /*W*/ "
                      "GROUP BY cancellation_reason ORDER BY cancelled DESC",
    "cancel_reason_city": "SELECT city, cancellation_reason, COUNT(*) AS cancelled FROM vw_orders /*W*/ "
                          "GROUP BY city, cancellation_reason",
    "scatter": "SELECT distance_km, delivery_time_min, delivery_delay_min FROM vw_orders /*W*/",
    "distance_bands": """SELECT distance_band, COUNT(*) AS orders, AVG(delivery_time_min) AS avg_time,
        AVG(delivery_delay_min) AS avg_delay, AVG(is_late)*100.0 AS late_pct
        FROM vw_orders /*W*/ GROUP BY distance_band""",
    "rating_vs_time": """SELECT delivery_time_band, COUNT(*) AS orders, AVG(delivery_rating) AS avg_rating
        FROM vw_orders /*W*/ GROUP BY delivery_time_band""",
    "perf_cat": "SELECT delivery_performance, COUNT(*) AS orders FROM vw_orders /*W*/ GROUP BY delivery_performance",
    "peak": f"SELECT peak_hour_flag, {KPI_COLS} FROM vw_orders /*W*/ GROUP BY peak_hour_flag",
    "restaurants": """SELECT restaurant_id, restaurant_name, COUNT(*) AS orders,
        AVG(restaurant_rating) AS avg_restaurant_rating, AVG(delivery_rating) AS avg_delivery_rating,
        AVG(is_cancelled)*100.0 AS cancel_rate,
        SUM(CASE WHEN order_status='Delivered' THEN final_amount END) AS revenue
        FROM vw_orders /*W*/ GROUP BY restaurant_id, restaurant_name""",
    "discount_amt": """SELECT discount_applied, COUNT(*) AS orders, AVG(final_amount) AS avg_final,
        AVG(@PROFIT@) AS profit_per_order, SUM(@PROFIT@) AS profit,
        SUM(@PROFIT@)*100.0/NULLIF(SUM(final_amount),0) AS margin
        FROM vw_orders /*W*/ GROUP BY discount_applied ORDER BY discount_applied""",
    "discount_band": """SELECT discount_band, COUNT(*) AS orders, AVG(@PROFIT@) AS profit_per_order,
        SUM(@PROFIT@)*100.0/NULLIF(SUM(final_amount),0) AS margin
        FROM vw_orders /*W*/ GROUP BY discount_band""",
    "city_cuisine": """SELECT city, cuisine_type, SUM(final_amount) AS revenue FROM vw_orders /*W*/
        GROUP BY city, cuisine_type""",
    "city_area_time": """SELECT city, area, AVG(delivery_time_min) AS avg_time FROM vw_orders /*W*/
        GROUP BY city, area""",
}

# ---- Ready-made analyst queries for the SQL Explorer (run on the FULL dataset) ----------
PRESET_QUERIES = {
    "Top 10 spending customers": """SELECT customer_id, COUNT(*) AS orders,
       ROUND(SUM(final_amount),0) AS total_spent, ROUND(AVG(final_amount),0) AS avg_order_value
FROM vw_orders WHERE order_status = 'Delivered'
GROUP BY customer_id ORDER BY total_spent DESC LIMIT 10;""",
    "Age group vs order value": """SELECT age_group, COUNT(*) AS orders, ROUND(AVG(final_amount),0) AS avg_order_value
FROM vw_orders WHERE order_status = 'Delivered' GROUP BY age_group ORDER BY avg_order_value DESC;""",
    "Weekday vs weekend patterns": """SELECT order_day_type, COUNT(*) AS orders, COUNT(DISTINCT order_date) AS days,
       ROUND(COUNT(*)*1.0/COUNT(DISTINCT order_date),1) AS orders_per_day,
       ROUND(AVG(final_amount),0) AS avg_order_value, ROUND(AVG(is_cancelled)*100,1) AS cancel_rate_pct
FROM vw_orders GROUP BY order_day_type;""",
    "Monthly revenue trend": """SELECT order_month, COUNT(*) AS orders,
       ROUND(SUM(CASE WHEN order_status='Delivered' THEN final_amount END),0) AS revenue
FROM vw_orders WHERE order_month IS NOT NULL GROUP BY order_month ORDER BY order_month;""",
    "Impact of discounts on profit": """SELECT discount_applied AS discount_rs, COUNT(*) AS orders,
       ROUND(AVG(profit_amount),2) AS avg_profit_per_order,
       ROUND(SUM(profit_amount)*100.0/SUM(final_amount),2) AS margin_pct
FROM vw_orders WHERE order_status='Delivered' GROUP BY discount_applied ORDER BY discount_applied;""",
    "Revenue by city": """SELECT city, ROUND(SUM(final_amount),0) AS revenue, COUNT(*) AS delivered_orders
FROM vw_orders WHERE order_status='Delivered' GROUP BY city ORDER BY revenue DESC;""",
    "Revenue by cuisine": """SELECT cuisine_type, ROUND(SUM(final_amount),0) AS revenue, COUNT(*) AS delivered_orders
FROM vw_orders WHERE order_status='Delivered' GROUP BY cuisine_type ORDER BY revenue DESC;""",
    "Average delivery time by city": """SELECT city, ROUND(AVG(delivery_time_min),1) AS avg_delivery_min,
       ROUND(AVG(delivery_rating),2) AS avg_rating
FROM vw_orders WHERE order_status='Delivered' GROUP BY city ORDER BY avg_delivery_min;""",
    "Distance vs delivery delay (observed rows only)": """SELECT distance_band, COUNT(*) AS orders,
       ROUND(AVG(delivery_time_min),1) AS avg_time_min, ROUND(AVG(delivery_delay_min),1) AS avg_delay_min,
       ROUND(AVG(is_late)*100,1) AS late_pct
FROM vw_orders WHERE distance_imputed=0 AND delivery_time_imputed=0
GROUP BY distance_band ORDER BY MIN(distance_km);""",
    "Delivery rating vs delivery time": """SELECT delivery_time_band, COUNT(*) AS orders, ROUND(AVG(delivery_rating),2) AS avg_rating
FROM vw_orders WHERE order_status='Delivered' AND delivery_time_imputed=0
GROUP BY delivery_time_band ORDER BY MIN(delivery_time_min);""",
    "Top-rated restaurants (min 50 orders)": """SELECT restaurant_id, restaurant_name, COUNT(*) AS orders,
       ROUND(AVG(restaurant_rating),2) AS avg_rating
FROM vw_orders GROUP BY restaurant_id, restaurant_name HAVING COUNT(*) >= 50
ORDER BY avg_rating DESC, orders DESC LIMIT 10;""",
    "Cancellation rate by restaurant (min 50 orders)": """SELECT restaurant_id, restaurant_name, COUNT(*) AS orders,
       ROUND(AVG(is_cancelled)*100,1) AS cancel_rate_pct
FROM vw_orders GROUP BY restaurant_id, restaurant_name HAVING COUNT(*) >= 50
ORDER BY cancel_rate_pct DESC LIMIT 10;""",
    "Cuisine-wise performance": """SELECT cuisine_type, COUNT(*) AS orders, ROUND(AVG(restaurant_rating),2) AS avg_restaurant_rating,
       ROUND(AVG(delivery_rating),2) AS avg_delivery_rating, ROUND(AVG(is_cancelled)*100,1) AS cancel_rate_pct,
       ROUND(AVG(CASE WHEN order_status='Delivered' THEN delivery_time_min END),1) AS avg_delivery_min
FROM vw_orders GROUP BY cuisine_type ORDER BY orders DESC;""",
    "Peak-hour demand": """SELECT peak_hour_flag, COUNT(*) AS orders, ROUND(AVG(final_amount),0) AS avg_order_value,
       ROUND(AVG(delivery_time_min),1) AS avg_delivery_min, ROUND(AVG(is_cancelled)*100,1) AS cancel_rate_pct
FROM vw_orders GROUP BY peak_hour_flag;""",
    "Payment-mode preference": """SELECT payment_mode, COUNT(*) AS orders,
       ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM vw_orders),1) AS share_pct
FROM vw_orders GROUP BY payment_mode ORDER BY orders DESC;""",
    "Cancellation reasons": """SELECT cancellation_reason, COUNT(*) AS cancelled,
       ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM vw_orders WHERE is_cancelled=1),1) AS share_pct
FROM vw_orders WHERE is_cancelled=1 GROUP BY cancellation_reason ORDER BY cancelled DESC;""",
}


# ----------------------------------------------------------------------------------
# 3. UI HELPERS
# ----------------------------------------------------------------------------------
def stretch(fn) -> dict:
    """Return the right 'full width' kwarg for the installed Streamlit version."""
    try:
        return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}
    except (TypeError, ValueError):
        return {"use_container_width": True}


_PLOT_N = [0]


def plot(fig: go.Figure) -> None:
    _PLOT_N[0] += 1
    st.plotly_chart(fig, key=f"pc{_PLOT_N[0]}", config={"displayModeBar": False}, **stretch(st.plotly_chart))


def table(df: pd.DataFrame, height: int | None = None) -> None:
    kw = {"height": height if height is not None else "content"}
    st.dataframe(
        df,
        hide_index=True,
        **kw,
        **stretch(st.dataframe)
    )


@contextmanager
def card(title: str, desc: str):
    """A bordered 'card' with a heading whose (i) icon shows a description on hover."""
    with st.container(border=True):
        st.subheader(title, help=desc, anchor=False)
        yield


def style(fig: go.Figure, h: int = 300, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=h, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#C9D1E6", size=12), showlegend=legend,
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=11)),
        hoverlabel=dict(bgcolor="#1B2447", font_color="#FFFFFF", bordercolor="#4C8DFF"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    return fig


def inr(x, compact: bool = True, dec: int = 2) -> str:
    if x is None or pd.isna(x):
        return "-"
    if compact and abs(x) >= 1e7:
        return f"₹{x / 1e7:,.{dec}f} Cr"
    if compact and abs(x) >= 1e5:
        return f"₹{x / 1e5:,.{dec}f} L"
    return f"₹{x:,.0f}"


def human(n) -> str:
    if n is None or pd.isna(n):
        return "-"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.1f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:,.0f}"


def order_by(df: pd.DataFrame, col: str, order: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=[o for o in order if o in set(df[col])], ordered=True)
    return df.sort_values(col).dropna(subset=[col]).assign(**{col: lambda d: d[col].astype(str)})


def month_labels(months: pd.Series) -> list[str]:
    d = pd.to_datetime(months + "-01")
    fmt = "%b" if d.dt.year.nunique() == 1 else "%b %y"
    return d.dt.strftime(fmt).tolist()


def donut(labels, values, center: str, colors=None, hover="Orders") -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62, sort=False, textinfo="percent",
        marker=dict(colors=colors or PALETTE, line=dict(color="#0B1020", width=2)),
        hovertemplate=f"<b>%{{label}}</b><br>{hover}: %{{value:,.0f}}<br>Share: %{{percent}}<extra></extra>"))
    fig.add_annotation(text=center, x=0.5, y=0.5, showarrow=False, font=dict(size=17, color="#FFFFFF"))
    style(fig, 290)
    fig.update_layout(legend=dict(orientation="v", y=0.5, x=1.0))
    return fig


def hbar(df, x, y, color=ORANGE, fmt=",.0f", prefix="", h=330, xtitle=None) -> go.Figure:
    d = df.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d[x], y=d[y], orientation="h", marker_color=color,
        text=[f"{prefix}{v:{fmt}}" for v in d[x]], textposition="outside", cliponaxis=False,
        hovertemplate=f"<b>%{{y}}</b><br>{xtitle or x}: {prefix}%{{x:{fmt}}}<extra></extra>"))
    fig.update_xaxes(title=xtitle)
    fig.update_layout(bargap=0.25)
    return style(fig, h, legend=False)


def vbar(df, x, y, color=BLUE, fmt=",.0f", prefix="", h=300, ytitle=None, colors=None) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=df[x], y=df[y], marker_color=colors or color,
        text=[f"{prefix}{v:{fmt}}" for v in df[y]], textposition="outside", cliponaxis=False,
        hovertemplate=f"<b>%{{x}}</b><br>{ytitle or y}: {prefix}%{{y:{fmt}}}<extra></extra>"))
    fig.update_yaxes(title=ytitle)
    fig.update_layout(bargap=0.3)
    return style(fig, h, legend=False)


def rev_bar(df: pd.DataFrame, col: str, color: str) -> go.Figure:
    d = df.sort_values("revenue", ascending=False)
    fig = go.Figure(go.Bar(
        x=d[col], y=d["revenue"] / 1e7, marker_color=color,
        text=[f"₹{v / 1e7:.2f} Cr" for v in d["revenue"]], textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:.2f} Cr<extra></extra>"))
    fig.update_yaxes(title="Revenue (₹ Cr)")
    return style(fig, 300, legend=False)


def metric_card(label, value, help_text, series=None, delta=None, inverse=False) -> None:
    kw = dict(label=label, value=value, help=help_text)
    if delta is not None:
        kw.update(delta=delta, delta_color="inverse" if inverse else "normal")
    full = dict(kw, border=True)
    if series is not None and len(series) > 1:
        full.update(chart_data=[float(v) for v in series], chart_type="area")
    try:
        st.metric(**full)
    except TypeError:                       # older Streamlit without sparkline / border support
        st.metric(**kw)


def pct_change(series: pd.Series) -> float | None:
    s = series.dropna()
    if len(s) < 2 or s.iloc[-2] in (0, None):
        return None
    return (s.iloc[-1] - s.iloc[-2]) / abs(s.iloc[-2]) * 100


CSS = """
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1700px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#0E1530 0%,#0B1020 100%);
    border-right: 1px solid rgba(255,255,255,.07);}
[data-testid="stSidebar"] .stButton button {justify-content: flex-start; border-radius: 10px;}
[data-testid="stMetric"] {background: linear-gradient(145deg,#121A38,#0F1630); border-radius: 14px;
    padding: .6rem .8rem; border: 1px solid rgba(255,255,255,.07);}
[data-testid="stMetricValue"] {font-size: 1.55rem; font-weight: 700;}
[data-testid="stMetricLabel"] p {font-size: .82rem; opacity: .85;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px;
    background: rgba(18,26,56,.55); border-color: rgba(255,255,255,.07);}
h3 {font-size: 1.02rem !important; font-weight: 650 !important; padding-bottom: .2rem !important;}
.hero {display:flex; justify-content:space-between; align-items:center; gap:1rem; padding:.2rem 0 .8rem 0;}
.hero h1 {margin:0; font-size:2rem; line-height:1.15;}
.hero p {margin:.15rem 0 0 0; opacity:.75;}
.badge {background:#121A38; border:1px solid rgba(255,255,255,.1); border-radius:12px;
    padding:.45rem .8rem; font-size:.78rem; line-height:1.25; white-space:nowrap;}
.badge b {display:block; font-size:.9rem;}
.brand {font-size:1.35rem; font-weight:800; margin-bottom:.1rem;}
.brandsub {font-size:.72rem; opacity:.65; margin-bottom:.8rem;}
</style>
"""

PAGES = {
    "Overview": ("🏠", "Executive summary: all KPIs plus the headline charts, exactly like the FoodDash mock-up."),
    "Customers & Orders": ("👥", "Who orders, how much they spend, and when: top spenders, age groups, gender, "
                                 "weekday vs weekend, order frequency."),
    "Revenue & Profit": ("💰", "Monthly revenue and profit trends, the impact of discounts on margin, and which "
                               "cities / cuisines earn the most."),
    "Delivery Performance": ("🚚", "Delivery speed by city, distance vs delay, rating vs delivery time, "
                                   "performance categories and peak vs off-peak."),
    "Restaurants & Cancellations": ("🍽️", "Top-rated restaurants, cancellation rate by restaurant, cuisine "
                                          "scorecards and why orders get cancelled."),
    "SQL Explorer": ("🗄️", "The SQL layer: schema, ER diagram, 16 ready-made analyst queries, and a read-only "
                           "query editor for your own SQL."),
    "Data Explorer": ("🔎", "Browse the cleaned rows, review the data-quality report and the cleaning log, and "
                            "download the filtered data."),
    "Documentation": ("📘", "Dataset, cleaning steps, feature engineering, database design, KPI definitions, "
                            "live findings, recommendations, and how to run / publish the project."),
}


# ----------------------------------------------------------------------------------
# 4. PAGES
# ----------------------------------------------------------------------------------
def kpi_row(f: dict) -> pd.DataFrame:
    k = run(SQL["kpi"], f).iloc[0]
    mon = run(SQL["monthly"], f, "order_month IS NOT NULL")
    specs = [
        ("Total Orders", "orders", human(k.orders) if k.orders >= 1e5 else f"{int(k.orders):,}",
         "Number of orders in the current filter (delivered + cancelled).", False),
        ("Total Revenue", "revenue", inr(k.revenue),
         "Sum of final_amount (after discount) on DELIVERED orders. Cancelled orders earn nothing.", False),
        ("Avg Order Value", "aov", inr(k.aov, compact=False),
         "Average final_amount of delivered orders = Revenue / Delivered orders.", False),
        ("Avg Delivery Time", "avg_delivery_time", f"{k.avg_delivery_time:.0f} min",
         "Average minutes from order to doorstep on delivered orders. Lower is better.", True),
        ("Cancellation Rate", "cancel_rate", f"{k.cancel_rate:.1f}%",
         "Cancelled orders / total orders. Lower is better.", True),
        ("Avg Delivery Rating", "avg_rating", f"{k.avg_rating:.2f} / 5",
         "Mean customer delivery rating (1-5). Only delivered orders can carry a rating.", False),
        ("Profit Margin", "margin", f"{k.margin:.1f}%",
         "Order-weighted margin = total profit / total revenue on delivered orders. Basis (corrected or "
         "raw) is set in the sidebar.", False),
    ]
    cols = st.columns(len(specs))
    for col, (label, key, val, hlp, inverse) in zip(cols, specs):
        with col:
            ch = pct_change(mon[key]) if len(mon) >= 2 else None
            metric_card(label, val, hlp + (" Arrow = last month vs the month before." if ch is not None else ""),
                        series=mon[key].tolist() if len(mon) > 1 else None,
                        delta=None if ch is None else f"{ch:+.1f}%", inverse=inverse)
    return mon


def chart_monthly(mon: pd.DataFrame) -> go.Figure:
    lab = month_labels(mon["order_month"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=lab, y=mon["orders"], name="Orders", marker_color=BLUE,
                         hovertemplate="<b>%{x}</b><br>Orders: %{y:,}<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=lab, y=mon["revenue"] / 1e7, name="Revenue (₹ Cr)", mode="lines+markers",
                             line=dict(color=PINK, width=3),
                             hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:.2f} Cr<extra></extra>"), secondary_y=True)
    fig.update_yaxes(title_text="Orders", secondary_y=False)
    fig.update_yaxes(title_text="Revenue (₹ Cr)", secondary_y=True, showgrid=False)
    return style(fig, 300)


def chart_city_map(city: pd.DataFrame, metric: str) -> go.Figure:
    cfg = {"Orders": ("orders", ",.0f", ""), "Revenue (₹ Cr)": ("revenue", ".2f", "₹"),
           "Avg delivery time (min)": ("avg_delivery_time", ".0f", ""),
           "Cancellation rate (%)": ("cancel_rate", ".1f", "")}
    col, fmt, pre = cfg[metric]
    d = city[city["city"].isin(CITY_COORDS)].copy()
    vals = d[col] / 1e7 if col == "revenue" else d[col]
    if d.empty:
        return style(go.Figure(), 300)
    span = (vals.max() - vals.min()) or 1
    size = 22 + 38 * (vals - vals.min()) / span
    fig = go.Figure(go.Scattergeo(
        lon=[CITY_COORDS[c][1] for c in d["city"]], lat=[CITY_COORDS[c][0] for c in d["city"]],
        text=[f"{c}<br>{pre}{v:{fmt}}" for c, v in zip(d["city"], vals)], mode="markers+text",
        textposition="top center", marker=dict(size=size, color=ORANGE, opacity=0.85,
                                                line=dict(color="#fff", width=1)),
        hovertemplate="<b>%{text}</b><extra></extra>", textfont=dict(color="#fff", size=11)))
    fig.update_geos(scope="asia", center=dict(lat=21, lon=79), projection_scale=4.3, showland=True,
                    landcolor="#18213f", showcountries=True, countrycolor="#3a4778", showocean=False,
                    showcoastlines=False, bgcolor="rgba(0,0,0,0)")
    return style(fig, 300, legend=False)


def chart_scatter(sc: pd.DataFrame, ycol="delivery_time_min", ylab="Delivery time (min)") -> go.Figure:
    corr = sc["distance_km"].corr(sc[ycol]) if len(sc) > 2 else float("nan")
    s = sc.sample(min(len(sc), 3000), random_state=1)
    fig = go.Figure(go.Scatter(x=s["distance_km"], y=s[ycol], mode="markers", name="Orders",
                               marker=dict(size=4, color=GREEN, opacity=0.35),
                               hovertemplate="Distance: %{x:.1f} km<br>" + ylab + ": %{y:.0f}<extra></extra>"))
    if len(sc) > 2:
        b, a = np.polyfit(sc["distance_km"], sc[ycol], 1)
        xs = np.array([sc["distance_km"].min(), sc["distance_km"].max()])
        fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", name="Trend",
                                 line=dict(color=PINK, width=3), hoverinfo="skip"))
    txt = ("n/a" if np.isnan(corr) else f"Correlation: {corr:.2f}")
    fig.add_annotation(text=txt, xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False,
                       bgcolor="#1B2447", bordercolor="#4C8DFF", font=dict(size=12), xanchor="left", yanchor="top")
    fig.update_xaxes(title="Distance (km)")
    fig.update_yaxes(title=ylab)
    return style(fig, 300, legend=False)


def page_overview(f: dict) -> None:
    mon = kpi_row(f)
    cui, city = run(SQL["by_cuisine"], f), run(SQL["by_city"], f)
    c1, c2, c3 = st.columns([5, 3.5, 4])
    with c1, card("Monthly Orders & Revenue Trend", "Bars = orders per month (left axis). Line = delivered "
                  "revenue in ₹ crore (right axis). Orders with an unknown date are excluded."):
        plot(chart_monthly(mon))
    with c2, card("Orders by Cuisine Type", "Share of orders per cuisine. 'Unknown' = cuisine missing in the "
                  "source data."):
        plot(donut(cui["cuisine_type"], cui["orders"], f"{human(cui['orders'].sum())}<br>Orders"))
    with c3, card("Orders by City", "Bubble size = selected metric per city. Switch the metric or view in the box "
                  "below. Orders whose city is missing are not shown on the map."):
        a, b = st.columns(2)
        metric = a.selectbox("Metric", ["Orders", "Revenue (₹ Cr)", "Avg delivery time (min)",
                                        "Cancellation rate (%)"], label_visibility="collapsed",
                             help="Choose what the bubbles represent.")
        view = b.radio("View", ["Map", "Bars"], horizontal=True, label_visibility="collapsed",
                       help="Map = bubbles on India map. Bars = ranked bar chart.")
        if view == "Map":
            plot(chart_city_map(city, metric))
        else:
            col = {"Orders": "orders", "Revenue (₹ Cr)": "revenue", "Avg delivery time (min)": "avg_delivery_time",
                   "Cancellation rate (%)": "cancel_rate"}[metric]
            d = city.sort_values(col, ascending=False)
            plot(vbar(d, "city", col, ORANGE, h=300, ytitle=metric))

    w, params = where_clause(f, DELIVERED)
    top = cached_query(CTX["sig"], SQL["top_customers"].replace("/*W*/", w), params + (10,))
    age = order_by(run(SQL["by_age"], f, DELIVERED), "age_group", AGE_ORDER)
    day = run(SQL["by_daytype"], f)
    c1, c2, c3 = st.columns([4.5, 4.5, 3])
    with c1, card("Top 10 Customers by Revenue", "The 10 customers whose delivered orders add up to the most "
                  "revenue (after discounts)."):
        plot(hbar(top, "revenue", "customer_id", ORANGE, ",.0f", "₹", 330, "Revenue"))
    with c2, card("Average Order Value by Age Group", "Average final amount per delivered order for each age "
                  "bracket. 'Unknown' = age missing in the source (not guessed)."):
        plot(vbar(age, "age_group", "aov", None, ",.0f", "₹", 330, "Avg order value",
                  colors=PALETTE[: len(age)]))
    with c3, card("Weekday vs Weekend Orders", "Share of orders on weekdays vs weekends. Weekends are only 2 of 7 "
                  "days, see Customers & Orders for the per-day view."):
        plot(donut(day["order_day_type"], day["orders"], f"{human(day['orders'].sum())}<br>Orders",
                   colors=[BLUE, ORANGE, GREY]))

    pay, reasons = run(SQL["payment"], f), run(SQL["cancel_reasons"], f, "is_cancelled = 1")
    w, params = where_clause(f, "distance_imputed = 0 AND delivery_time_imputed = 0")
    sc = cached_query(CTX["sig"], SQL["scatter"].replace("/*W*/", w), params)
    c1, c2, c3, c4 = st.columns(4)
    with c1, card("Delivery Performance by City", "Bars = average delivery time in minutes (left). Line = average "
                  "delivery rating (right). Delivered orders only."):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        d = run(SQL["by_city"], f, DELIVERED)
        fig.add_trace(go.Bar(x=d["city"], y=d["avg_delivery_time"], name="Avg time (min)", marker_color=BLUE,
                             hovertemplate="<b>%{x}</b><br>Avg time: %{y:.0f} min<extra></extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(x=d["city"], y=d["avg_rating"], name="Avg rating", mode="lines+markers",
                                 line=dict(color=PINK, width=3),
                                 hovertemplate="<b>%{x}</b><br>Rating: %{y:.2f}<extra></extra>"), secondary_y=True)
        fig.update_yaxes(range=[0, 5], secondary_y=True, showgrid=False)
        plot(style(fig, 280))
    with c2, card("Distance vs Delivery Time", "Each dot is one order where BOTH distance and delivery time were "
                  "actually recorded (imputed values are excluded so the correlation is honest)."):
        plot(chart_scatter(sc))
    with c3, card("Payment Mode Preference", "Share of orders by payment method. 'Unknown' = payment mode "
                  "missing in the source data."):
        plot(donut(pay["payment_mode"], pay["orders"], f"{human(pay['orders'].sum())}<br>Orders",
                   colors=[BLUE, ORANGE, GREEN, PURPLE, GREY]))
    with c4, card("Cancellation Reasons", "Why cancelled orders were cancelled. 'Not Specified' = the source had "
                  "no reason recorded."):
        reasons["share"] = reasons["cancelled"] / max(reasons["cancelled"].sum(), 1) * 100
        plot(hbar(reasons, "share", "cancellation_reason", RED, ".1f", "", 280, "Share of cancellations (%)"))


def page_customers(f: dict) -> None:
    st.caption("All numbers follow the sidebar filters.")
    n = st.slider("How many top customers?", 5, 25, 10, help="Number of customers shown in the ranking.")
    w, params = where_clause(f, DELIVERED)
    top = cached_query(CTX["sig"], SQL["top_customers"].replace("/*W*/", w), params + (n,))
    c1, c2 = st.columns([3, 2])
    with c1, card(f"Top {n} Spending Customers", "Customers ranked by total delivered revenue. Orders = all their "
                  "orders in the filter."):
        plot(hbar(top, "revenue", "customer_id", ORANGE, ",.0f", "₹", max(300, 28 * n), "Revenue"))
    with c2, card("Top Customer Table", "Same ranking with order counts, average order value and cancellation rate."):
        t = top.rename(columns={"customer_id": "Customer", "orders": "Orders", "revenue": "Revenue ₹",
                                "aov": "AOV ₹", "cancel_rate": "Cancel %"}).round(1)
        table(t, height=max(300, 35 * n))
    age = order_by(run(SQL["by_age"], f, DELIVERED), "age_group", AGE_ORDER)
    gen = run(SQL["by_gender"], f, DELIVERED)
    c1, c2, c3 = st.columns(3)
    with c1, card("Order Value by Age Group", "Average order value for each age bracket (delivered orders)."):
        plot(vbar(age, "age_group", "aov", None, ",.0f", "₹", 300, "Avg order value", colors=PALETTE[: len(age)]))
    with c2, card("Orders by Age Group", "How many delivered orders each age bracket placed."):
        plot(vbar(age, "age_group", "orders", None, ",.0f", "", 300, "Orders", colors=PALETTE[: len(age)]))
    with c3, card("Gender Split", "Delivered orders and average order value by customer gender."):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=gen["customer_gender"], y=gen["orders"], name="Orders", marker_color=BLUE,
                             hovertemplate="<b>%{x}</b><br>Orders: %{y:,}<extra></extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(x=gen["customer_gender"], y=gen["aov"], name="AOV ₹", mode="lines+markers",
                                 line=dict(color=PINK, width=3),
                                 hovertemplate="<b>%{x}</b><br>AOV: ₹%{y:,.0f}<extra></extra>"), secondary_y=True)
        fig.update_yaxes(showgrid=False, secondary_y=True)
        plot(style(fig, 300))
    day = run(SQL["by_daytype"], f)
    day = day[day["order_day_type"] != "Unknown"].copy()
    day["orders_per_day"] = day["orders"] / day["days"].clip(lower=1)
    opc = run(SQL["orders_per_customer"], f)
    c1, c2, c3 = st.columns(3)
    with c1, card("Weekday vs Weekend: Orders per Day", "Total orders divided by the number of distinct days of "
                  "that type, a fair comparison because weekends have fewer days."):
        plot(vbar(day, "order_day_type", "orders_per_day", None, ",.0f", "", 300, "Orders per day",
                  colors=[BLUE, ORANGE][: len(day)]))
    with c2, card("Weekday vs Weekend: Value & Cancellations", "Average order value (bars) and cancellation rate "
                  "(line) by day type."):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=day["order_day_type"], y=day["aov"], name="AOV ₹", marker_color=ORANGE,
                             hovertemplate="<b>%{x}</b><br>AOV: ₹%{y:,.0f}<extra></extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(x=day["order_day_type"], y=day["cancel_rate"], name="Cancel %", mode="lines+markers",
                                 line=dict(color=RED, width=3),
                                 hovertemplate="<b>%{x}</b><br>Cancel: %{y:.1f}%<extra></extra>"), secondary_y=True)
        fig.update_yaxes(showgrid=False, secondary_y=True)
        plot(style(fig, 300))
    with c3, card("Order Frequency", "How many customers placed 1, 2, 3 ... orders in the filter (repeat behaviour)."):
        fig = go.Figure(go.Bar(x=opc["orders_per_customer"], y=opc["customers"], marker_color=GREEN,
                               hovertemplate="%{x} orders: %{y:,} customers<extra></extra>"))
        fig.update_xaxes(title="Orders per customer")
        fig.update_yaxes(title="Customers")
        plot(style(fig, 300, legend=False))


def page_revenue(f: dict) -> None:
    mon = run(SQL["monthly"], f, "order_month IS NOT NULL")
    lab = month_labels(mon["order_month"]) if len(mon) else []
    c1, c2 = st.columns([3, 2])
    with c1, card("Monthly Revenue & Profit Margin", "Bars = delivered revenue (₹ lakh). Line = order-weighted "
                  "profit margin %. Basis follows the sidebar setting."):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=lab, y=mon["revenue"] / 1e5, name="Revenue (₹ L)", marker_color=BLUE,
                             hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.1f} L<extra></extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(x=lab, y=mon["margin"], name="Margin %", mode="lines+markers",
                                 line=dict(color=GREEN, width=3),
                                 hovertemplate="<b>%{x}</b><br>Margin: %{y:.1f}%<extra></extra>"), secondary_y=True)
        fig.update_yaxes(showgrid=False, secondary_y=True)
        plot(style(fig, 320))
    with c2, card("Monthly Table", "Month-by-month orders, revenue, AOV and margin."):
        t = mon[["order_month", "orders", "revenue", "aov", "margin"]].copy()
        t.columns = ["Month", "Orders", "Revenue ₹", "AOV ₹", "Margin %"]
        table(t.round(1), height=320)

    da, db = run(SQL["discount_amt"], f, DELIVERED), order_by(run(SQL["discount_band"], f, DELIVERED),
                                                              "discount_band", DISC_PCT_ORDER)
    c1, c2 = st.columns(2)
    with c1, card("Impact of Discounts on Profit (by ₹ discount)", "Bars = average profit per order at each "
                  "discount amount. Line = margin %. If bars shrink as discount grows, discounts are eating profit."):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=[f"₹{int(x)}" for x in da["discount_applied"]], y=da["profit_per_order"],
                             name="Profit / order ₹", marker_color=ORANGE,
                             hovertemplate="<b>Discount %{x}</b><br>Profit/order: ₹%{y:,.0f}<extra></extra>"),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=[f"₹{int(x)}" for x in da["discount_applied"]], y=da["margin"], name="Margin %",
                                 mode="lines+markers", line=dict(color=GREEN, width=3),
                                 hovertemplate="<b>Discount %{x}</b><br>Margin: %{y:.1f}%<extra></extra>"),
                      secondary_y=True)
        fig.update_yaxes(showgrid=False, secondary_y=True)
        fig.update_xaxes(title="Discount applied")
        plot(style(fig, 320))
    with c2, card("Impact of Discounts on Profit (by % of order)", "Same idea, but discounts are grouped by how "
                  "large they were relative to the order value."):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=db["discount_band"], y=db["profit_per_order"], name="Profit / order ₹",
                             marker_color=PURPLE,
                             hovertemplate="<b>%{x}</b><br>Profit/order: ₹%{y:,.0f}<extra></extra>"),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=db["discount_band"], y=db["margin"], name="Margin %", mode="lines+markers",
                                 line=dict(color=GREEN, width=3),
                                 hovertemplate="<b>%{x}</b><br>Margin: %{y:.1f}%<extra></extra>"), secondary_y=True)
        fig.update_yaxes(showgrid=False, secondary_y=True)
        fig.update_xaxes(title="Discount as % of order value")
        plot(style(fig, 320))

    city, cui = run(SQL["by_city"], f), run(SQL["by_cuisine"], f)
    cc = run(SQL["city_cuisine"], f, DELIVERED)
    c1, c2 = st.columns(2)
    with c1, card("Revenue by City", "Delivered revenue per city, highest first."):
        plot(rev_bar(city, "city", ORANGE))
    with c2, card("Revenue by Cuisine", "Delivered revenue per cuisine, highest first."):
        plot(rev_bar(cui, "cuisine_type", PURPLE))
    with card("City × Cuisine Revenue Heatmap", "Where revenue concentrates. Darker/brighter = more revenue "
              "(₹ crore). Good for spotting under-served city-cuisine pairs."):
        pv = cc.pivot_table(index="city", columns="cuisine_type", values="revenue", aggfunc="sum") / 1e7
        fig = go.Figure(go.Heatmap(z=pv.values, x=list(pv.columns), y=list(pv.index), colorscale="Viridis",
                                   hovertemplate="<b>%{y} · %{x}</b><br>₹%{z:.2f} Cr<extra></extra>",
                                   colorbar=dict(title="₹ Cr")))
        plot(style(fig, 330, legend=False))


def page_delivery(f: dict) -> None:
    k = run(SQL["kpi"], f, DELIVERED).iloc[0]
    late = run("SELECT AVG(is_late)*100.0 AS late_pct, AVG(distance_km) AS dist FROM vw_orders /*W*/", f, DELIVERED).iloc[0]
    c = st.columns(4)
    with c[0]:
        metric_card("Avg Delivery Time", f"{k.avg_delivery_time:.0f} min", "Average delivery time, delivered orders.")
    with c[1]:
        metric_card("Late Deliveries", f"{late.late_pct:.1f}%",
                    f"Share of delivered orders slower than the expected time = {BASE_PREP_MIN:.0f} min + "
                    f"{MIN_PER_KM} min per km.")
    with c[2]:
        metric_card("Avg Distance", f"{late.dist:.1f} km", "Average delivery distance (missing values were filled "
                    "with the median).")
    with c[3]:
        metric_card("Avg Delivery Rating", f"{k.avg_rating:.2f} / 5", "Mean delivery rating on delivered orders.")

    city = run(SQL["by_city"], f, DELIVERED)
    ca = run(SQL["city_area_time"], f, DELIVERED)
    c1, c2 = st.columns(2)
    with c1, card("Average Delivery Time by City", "Mean delivery minutes per city (delivered orders)."):
        d = city.sort_values("avg_delivery_time")
        plot(vbar(d, "city", "avg_delivery_time", BLUE, ".0f", "", 300, "Avg delivery time (min)"))
    with c2, card("City × Area Delivery Time", "Average delivery minutes for every city/area pair. Brighter = "
                  "slower. Helps target the worst zones."):
        pv = ca.pivot_table(index="city", columns="area", values="avg_time")
        fig = go.Figure(go.Heatmap(z=pv.values, x=list(pv.columns), y=list(pv.index), colorscale="Inferno",
                                   hovertemplate="<b>%{y} · %{x}</b><br>%{z:.0f} min<extra></extra>",
                                   colorbar=dict(title="min")))
        plot(style(fig, 300, legend=False))

    w, params = where_clause(f, "distance_imputed = 0 AND delivery_time_imputed = 0")
    sc = cached_query(CTX["sig"], SQL["scatter"].replace("/*W*/", w), params)
    bands = order_by(cached_query(CTX["sig"], SQL["distance_bands"].replace("/*W*/", w), params),
                     "distance_band", DISTANCE_ORDER)
    c1, c2 = st.columns(2)
    with c1, card("Distance vs Delivery Delay", "Each dot = one order with recorded distance and time. Delay = "
                  "actual time minus expected time. The trend line and correlation show whether distance "
                  "really drives delay."):
        plot(chart_scatter(sc, "delivery_delay_min", "Delay vs expected (min)"))
    with c2, card("Delay by Distance Band", "Bars = average delay (minutes, negative = faster than expected). "
                  "Line = % of orders that were late."):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=bands["distance_band"], y=bands["avg_delay"], name="Avg delay (min)",
                             marker_color=ORANGE,
                             hovertemplate="<b>%{x}</b><br>Avg delay: %{y:.0f} min<extra></extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(x=bands["distance_band"], y=bands["late_pct"], name="Late %", mode="lines+markers",
                                 line=dict(color=RED, width=3),
                                 hovertemplate="<b>%{x}</b><br>Late: %{y:.1f}%<extra></extra>"), secondary_y=True)
        fig.update_yaxes(showgrid=False, secondary_y=True)
        plot(style(fig, 300))

    w2, p2 = where_clause(f, f"{DELIVERED} AND delivery_time_imputed = 0")
    rt = order_by(cached_query(CTX["sig"], SQL["rating_vs_time"].replace("/*W*/", w2), p2),
                  "delivery_time_band", TIME_BAND_ORDER)
    perf = order_by(run(SQL["perf_cat"], f, DELIVERED), "delivery_performance", PERF_ORDER)
    peak = run(SQL["peak"], f)
    c1, c2, c3 = st.columns(3)
    with c1, card("Delivery Rating vs Delivery Time", "Average rating for each delivery-time band. A falling "
                  "line means slow orders hurt customer satisfaction."):
        fig = go.Figure(go.Scatter(x=rt["delivery_time_band"], y=rt["avg_rating"], mode="lines+markers+text",
                                   text=[f"{v:.2f}" for v in rt["avg_rating"]], textposition="top center",
                                   line=dict(color=PINK, width=3),
                                   hovertemplate="<b>%{x}</b><br>Avg rating: %{y:.2f}<extra></extra>"))
        fig.update_yaxes(range=[0, 5], title="Avg rating")
        plot(style(fig, 300, legend=False))
    with c2, card("Delivery Performance Categories", "Fast <=45 min, Standard 46-90, Slow 91-150, Very Slow >150."):
        plot(donut(perf["delivery_performance"], perf["orders"], f"{human(perf['orders'].sum())}<br>Delivered",
                   colors=[GREEN, BLUE, ORANGE, RED]))
    with c3, card("Peak vs Off-Peak Demand", "Orders (bars) and average delivery time (line) by peak-hour flag. "
                  "'Unknown' = flag missing in the source."):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=peak["peak_hour_flag"], y=peak["orders"], name="Orders", marker_color=PURPLE,
                             hovertemplate="<b>%{x}</b><br>Orders: %{y:,}<extra></extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(x=peak["peak_hour_flag"], y=peak["avg_delivery_time"], name="Avg time (min)",
                                 mode="lines+markers", line=dict(color=YELLOW, width=3),
                                 hovertemplate="<b>%{x}</b><br>%{y:.0f} min<extra></extra>"), secondary_y=True)
        fig.update_yaxes(showgrid=False, secondary_y=True)
        plot(style(fig, 300))
    st.caption("The source file's order_time is always 00:00, so an hour-by-hour demand curve is impossible; "
               "the supplied Peak_Hour flag is the only peak-hour signal available.")


def page_restaurants(f: dict) -> None:
    rest = run(SQL["restaurants"], f)
    mx = int(rest["orders"].max()) if len(rest) else 1
    mn = st.slider("Minimum orders per restaurant", 1, max(mx, 2), min(50, max(mx, 2)),
                   help="Restaurants with fewer orders are hidden so a single lucky/unlucky order cannot top "
                        "or bottom the ranking.")
    r = rest[rest["orders"] >= mn].copy()
    r["label"] = r["restaurant_id"] + " · " + r["restaurant_name"].fillna("")
    c1, c2 = st.columns(2)
    with c1, card("Top 10 Rated Restaurants", "Highest average restaurant rating among restaurants that meet the "
                  "minimum-order threshold (ties broken by order count)."):
        top = r.sort_values(["avg_restaurant_rating", "orders"], ascending=False).head(10)
        plot(hbar(top, "avg_restaurant_rating", "label", GREEN, ".2f", "", 330, "Avg rating"))
    with c2, card("Highest Cancellation Rate: Top 10 Restaurants", "Restaurants with the most cancelled orders as "
                  "a share of their orders (min-order threshold applies)."):
        worst = r.sort_values(["cancel_rate", "orders"], ascending=False).head(10)
        plot(hbar(worst, "cancel_rate", "label", RED, ".1f", "", 330, "Cancel rate (%)"))
    with st.expander("Full restaurant table (searchable, sortable)"):
        t = r[["restaurant_id", "restaurant_name", "orders", "avg_restaurant_rating", "avg_delivery_rating",
               "cancel_rate", "revenue"]].round(2)
        t.columns = ["ID", "Name (most common)", "Orders", "Avg restaurant rating", "Avg delivery rating",
                     "Cancel %", "Revenue ₹"]
        table(t.sort_values("Orders", ascending=False), height=360)

    cui = run(SQL["by_cuisine"], f)
    rating = run("SELECT cuisine_type, AVG(restaurant_rating) AS avg_restaurant_rating FROM vw_orders /*W*/ "
                 "GROUP BY cuisine_type", f)
    cui = cui.merge(rating, on="cuisine_type")
    c1, c2 = st.columns([3, 2])
    with c1, card("Cuisine-wise Performance", "Pick a metric to compare cuisines."):
        mname = st.selectbox("Metric", ["Orders", "Revenue (₹ Cr)", "Avg restaurant rating", "Avg delivery rating",
                                        "Cancellation rate (%)", "Avg delivery time (min)", "Profit margin (%)"],
                             label_visibility="collapsed", help="Metric used for the bars.")
        col = {"Orders": "orders", "Revenue (₹ Cr)": "revenue", "Avg restaurant rating": "avg_restaurant_rating",
               "Avg delivery rating": "avg_rating", "Cancellation rate (%)": "cancel_rate",
               "Avg delivery time (min)": "avg_delivery_time", "Profit margin (%)": "margin"}[mname]
        d = cui.sort_values(col, ascending=False)
        y = d[col] / 1e7 if col == "revenue" else d[col]
        plot(style(go.Figure(go.Bar(x=d["cuisine_type"], y=y, marker_color=PALETTE[: len(d)],
                                    text=[f"{v:,.2f}" for v in y], textposition="outside", cliponaxis=False,
                                    hovertemplate="<b>%{x}</b><br>" + mname + ": %{y:,.2f}<extra></extra>")),
                   300, False))
    with c2, card("Cuisine Scorecard", "All cuisine metrics side by side."):
        t = cui[["cuisine_type", "orders", "revenue", "avg_restaurant_rating", "avg_rating", "cancel_rate",
                 "avg_delivery_time", "margin"]].copy()
        t["revenue"] = t["revenue"] / 1e7
        t.columns = ["Cuisine", "Orders", "Revenue ₹Cr", "Rest. rating", "Deliv. rating", "Cancel %", "Time min",
                     "Margin %"]
        table(t.round(2), height=300)

    reasons = run(SQL["cancel_reasons"], f, "is_cancelled = 1")
    crc = run(SQL["cancel_reason_city"], f, "is_cancelled = 1")
    pay = run(SQL["payment"], f)
    peak = run(SQL["peak"], f)
    c1, c2, c3 = st.columns(3)
    with c1, card("Cancellation Reasons", "Number of cancelled orders by recorded reason."):
        plot(donut(reasons["cancellation_reason"], reasons["cancelled"],
                   f"{human(reasons['cancelled'].sum())}<br>Cancelled",
                   colors=[RED, ORANGE, BLUE, GREY], hover="Cancelled"))
    with c2, card("Cancellations by City & Reason", "Stacked bars: which reasons dominate in each city."):
        fig = go.Figure()
        for i, rs in enumerate(sorted(crc["cancellation_reason"].unique())):
            d = crc[crc["cancellation_reason"] == rs]
            fig.add_trace(go.Bar(x=d["city"], y=d["cancelled"], name=rs, marker_color=PALETTE[i % len(PALETTE)],
                                 hovertemplate="<b>%{x}</b><br>" + rs + ": %{y:,}<extra></extra>"))
        fig.update_layout(barmode="stack")
        plot(style(fig, 290))
    with c3, card("Cancel Rate by Payment Mode & Peak Flag", "Is cancellation tied to how people pay, or to "
                  "peak hours? Bars show the cancellation rate (%)."):
        a = pay[["payment_mode", "cancel_rate"]].rename(columns={"payment_mode": "g"})
        b = peak[["peak_hour_flag", "cancel_rate"]].rename(columns={"peak_hour_flag": "g"})
        b["g"] = "Hour: " + b["g"]
        d = pd.concat([a, b])
        plot(vbar(d, "g", "cancel_rate", None, ".1f", "", 290, "Cancel rate (%)",
                  colors=[BLUE] * len(a) + [PURPLE] * len(b)))


def page_sql(f: dict) -> None:
    t1, t2, t3 = st.tabs(["🧪 Query runner", "🧱 Schema & ER diagram", "🔌 Optional: MySQL export"])
    with t1:
        st.markdown("Every KPI and chart in this app is a SQL query on the view **`vw_orders`**. "
                    "Pick a ready-made analyst query or write your own (read-only, full dataset, max 5,000 rows).")
        name = st.selectbox("Ready-made analyst query", list(PRESET_QUERIES),
                            help="These answer the project's 'Analyst Tasks'. Edit the SQL below if you like.")
        sql = st.text_area("SQL", value=PRESET_QUERIES[name], height=190, key=f"sql_{name}",
                           help="Only SELECT / WITH statements are allowed; the database is opened read-only.")
        if st.button("▶ Run query", type="primary", help="Execute the SQL above against the SQLite database.",
                     **stretch(st.button)):
            q = sql.strip().rstrip(";").strip()
            if not re.match(r"(?is)^(select|with)\b", q):
                st.error("Only SELECT / WITH queries are allowed here.")
            else:
                try:
                    res = sql_df(f"SELECT * FROM ({q}) LIMIT 5000")
                    st.success(f"{len(res):,} row(s)")
                    table(res)
                    if len(res):
                        st.download_button("⬇ Download result as CSV", res.to_csv(index=False).encode(),
                                           "query_result.csv", "text/csv", help="Save this result table.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"SQL error: {e}")
    with t2:
        st.graphviz_chart(ERD_DOT)
        st.caption("Star schema: one fact table (`orders`) + dimension tables. `vw_orders` joins them back into a "
                   "flat, analysis-friendly view.")
        tabs = sql_df("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY type DESC, name")
        sel = st.selectbox("Inspect a table / view", tabs["name"].tolist(),
                           help="Shows the columns and data types.")
        con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
        info = pd.read_sql_query(f"PRAGMA table_info('{sel}')", con)[["name", "type", "pk", "notnull"]]
        cnt = con.execute(f"SELECT COUNT(*) FROM '{sel}'").fetchone()[0]
        con.close()
        st.write(f"**{sel}**: {cnt:,} rows")
        table(info)
        with st.expander("Full DDL (CREATE TABLE statements)"):
            st.code(SCHEMA_SQL, language="sql")
    with t3:
        st.markdown("Your project brief mentions MySQL. The app already works with SQLite (zero setup). If you "
                    "also want the tables in MySQL, install the optional drivers "
                    "(`pip install sqlalchemy pymysql`) and push the tables from here.")
        st.warning("This **replaces** tables of the same names in the target database. It is optional and not "
                   "needed to use the dashboard.")
        url = st.text_input("SQLAlchemy URL", placeholder="mysql+pymysql://user:password@localhost:3306/food_db",
                            type="password", help="Create the empty database first: CREATE DATABASE food_db;")
        ok = st.checkbox("I understand existing tables will be replaced")
        if st.button("Push tables to MySQL", disabled=not (url and ok), help="Copies all 8 tables."):
            try:
                from sqlalchemy import create_engine
                eng = create_engine(url)
                for t in ["dim_city", "dim_area", "dim_cuisine", "dim_payment_mode", "dim_customer",
                          "dim_restaurant", "dim_delivery_partner", "orders"]:
                    sql_df(f"SELECT * FROM {t}").to_sql(t, eng, if_exists="replace", index=False, chunksize=5000)
                st.success("Done. Tables created in MySQL (types are inferred by pandas; add keys manually "
                           "if you need strict constraints).")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not push to MySQL: {e}")


def page_data(f: dict) -> None:
    meta = get_meta(CTX["sig"])
    t1, t2, t3 = st.tabs(["📄 Data preview", "🩺 Data-quality report", "🧹 Cleaning log"])
    with t1:
        q = st.text_input("Search customer / order / restaurant ID", help="Case-insensitive match, e.g. CUST1239 or ORD00012.")
        extra = None
        if q.strip():
            extra = "(customer_id LIKE ? OR order_id LIKE ? OR restaurant_id LIKE ?)"
        w, params = where_clause(f, extra)
        if q.strip():
            params = params + (f"%{q.strip()}%",) * 3
        total = cached_query(CTX["sig"], f"SELECT COUNT(*) AS n FROM vw_orders {w}", params)["n"][0]
        rows = st.slider("Rows to preview", 100, 5000, 500, 100, help="Preview size (the download has all rows).")
        prev = cached_query(CTX["sig"], f"SELECT * FROM vw_orders {w} LIMIT {int(rows)}", params)
        st.caption(f"{total:,} rows match the current filters; showing the first {len(prev):,}.")
        table(prev, height=420)
        if st.button("Prepare filtered CSV for download", help="Builds a CSV of every row that matches the filters."):
            full = cached_query(CTX["sig"], f"SELECT * FROM vw_orders {w}", params)
            st.download_button(f"⬇ Download {len(full):,} rows", full.to_csv(index=False).encode(),
                               "filtered_orders.csv", "text/csv")
    with t2:
        qual = pd.DataFrame(json.loads(meta["quality"]))
        st.caption(f"Raw rows: {int(meta['raw_rows']):,}  →  clean rows: {int(meta['clean_rows']):,}. "
                   "'Missing after' counts columns that remain empty on purpose (e.g. dates are never guessed; "
                   "cancelled orders have no rating).")
        table(qual)
        miss = qual.sort_values("Missing % before", ascending=False)
        miss = miss[miss["Missing % before"] > 0]
        if len(miss):
            plot(vbar(miss, "Column", "Missing % before", ORANGE, ".1f", "", 300, "% missing in raw file"))
    with t3:
        table(pd.DataFrame(json.loads(meta["cleaning_log"])), height=520)


def live_findings() -> list[str]:
    nf = {"cities": None, "cuisines": None, "statuses": None, "date_from": None, "date_to": None,
          "profit_col": "profit_amount"}
    out = []
    k = run(SQL["kpi"], nf).iloc[0]
    out.append(f"**Scale:** {int(k.orders):,} orders from {int(k.customers):,} customers generated "
               f"{inr(k.revenue)} of delivered revenue (AOV {inr(k.aov, False)}); overall margin "
               f"{k.margin:.1f}% (corrected basis).")
    city = run(SQL["by_city"], nf, DELIVERED).sort_values("revenue", ascending=False)
    city = city[city["city"] != "Unknown"]
    spread = (city["revenue"].max() / city["revenue"].min() - 1) * 100
    out.append(f"**Cities:** {city.iloc[0]['city']} leads revenue ({inr(city.iloc[0]['revenue'])}) and "
               f"{city.iloc[-1]['city']} trails; the gap between best and worst is only {spread:.0f}%, "
               "so demand is very evenly spread.")
    cui = run(SQL["by_cuisine"], nf, DELIVERED + " AND cuisine_type <> 'Unknown'").sort_values("revenue", ascending=False)
    out.append(f"**Cuisines:** {cui.iloc[0]['cuisine_type']} earns the most ({inr(cui.iloc[0]['revenue'])}); "
               f"{cui.iloc[-1]['cuisine_type']} the least.")
    mon = run(SQL["monthly"], nf, "order_month IS NOT NULL")
    hi, lo = mon.loc[mon["orders"].idxmax()], mon.loc[mon["orders"].idxmin()]
    out.append(f"**Seasonality:** busiest month {hi['order_month']} ({int(hi['orders']):,} orders), quietest "
               f"{lo['order_month']} ({int(lo['orders']):,}).")
    rs = run(SQL["cancel_reasons"], nf, "is_cancelled = 1")
    out.append(f"**Cancellations:** {k.cancel_rate:.1f}% of orders are cancelled. "
               f"'{rs.iloc[0]['cancellation_reason']}' is the largest recorded reason "
               f"({rs.iloc[0]['cancelled'] / rs['cancelled'].sum() * 100:.0f}% of cancellations).")
    sc = run(SQL["scatter"], nf, "distance_imputed = 0 AND delivery_time_imputed = 0")
    r = sc["distance_km"].corr(sc["delivery_time_min"])
    verdict = ("no meaningful relationship" if abs(r) < 0.1 else "a weak relationship" if abs(r) < 0.4
               else "a clear positive relationship")
    out.append(f"**Distance vs delivery time:** correlation r = {r:.2f} on {len(sc):,} orders with both values "
               f"recorded, which indicates {verdict}. Delivery time is driven by something other than distance "
               "(preparation time, rider allocation, traffic).")
    da = run(SQL["discount_amt"], nf, DELIVERED)
    out.append(f"**Discounts:** margin is {da.iloc[0]['margin']:.1f}% with no discount and "
               f"{da.iloc[-1]['margin']:.1f}% at ₹{int(da.iloc[-1]['discount_applied'])} off; profit per order "
               f"moves from ₹{da.iloc[0]['profit_per_order']:,.0f} to ₹{da.iloc[-1]['profit_per_order']:,.0f}.")
    dt_ = run(SQL["by_daytype"], nf)
    dt_ = dt_[dt_["order_day_type"] != "Unknown"].assign(opd=lambda d: d["orders"] / d["days"])
    if len(dt_) == 2:
        wd, we = dt_.set_index("order_day_type").loc["Weekday", "opd"], dt_.set_index("order_day_type").loc["Weekend", "opd"]
        out.append(f"**Weekday vs weekend:** {wd:,.0f} orders/day on weekdays vs {we:,.0f} on weekends "
                   f"({(we / wd - 1) * 100:+.0f}%).")
    return out


DOC_ABOUT = """
### Project
**Online Food Delivery Analysis: Data-Driven Business Insights.** A capstone-style analytics project on
100,000 online food-delivery orders (25 raw columns) covering customers, restaurants, orders, payments and
delivery performance.

### Goals
- Understand customer ordering behaviour
- Find operational inefficiencies in delivery
- Evaluate restaurant performance
- Track revenue, profit and cancellations
- Give management a KPI dashboard for decisions on pricing, discounts and logistics

### How the app is organised
| Layer | What it does |
|---|---|
| **Python / pandas** | Loads and cleans the CSV, engineers features (`clean_data`) |
| **SQL (SQLite)** | Stores a normalised star schema + the view `vw_orders`; all KPIs and charts are SQL queries |
| **Cache** | DB built once into `data/processed/`; later starts are instant |
| **Streamlit + Plotly** | Dashboard, filters, SQL explorer, data explorer, documentation |
"""

DOC_DICT = pd.DataFrame([
    ("order_id", "Raw", "Unique order key (100,000 unique)"),
    ("customer_id / customer_age / customer_gender", "Raw", "Customer; age 18-60; gender Male/Female/Other"),
    ("city / area", "Raw", "5 cities, 5 areas (North/South/East/West/Central)"),
    ("restaurant_id / restaurant_name", "Raw", "900 IDs, 500 names (see caveats)"),
    ("cuisine_type", "Raw", "Indian, Chinese, Italian, Mexican, Arabian"),
    ("order_date / order_time", "Raw", "2024 dates; time is always 00:00"),
    ("delivery_time_min / distance_km", "Raw", "Minutes 20-300; km 1-40"),
    ("order_value / discount_applied / final_amount", "Raw", "₹; discount is an absolute ₹ amount"),
    ("payment_mode / order_status / cancellation_reason", "Raw", "UPI, COD, Card, Wallet; Delivered/Cancelled"),
    ("delivery_partner_id / delivery_rating / restaurant_rating", "Raw", "Ratings 1-5 (restaurant had up to 5.5)"),
    ("order_day / peak_hour / profit_margin", "Raw", "Weekday/Weekend; True/False; fraction -0.2 to 0.5"),
    ("order_day_type", "Derived", "Weekday / Weekend, recomputed from the date"),
    ("peak_hour_flag, is_peak_hour", "Derived", "Peak / Off-Peak / Unknown from the supplied flag"),
    ("profit_margin_pct, profit_margin_raw_pct", "Derived", "Margin in %, floored at 0 (corrected) and as supplied (raw)"),
    ("profit_amount, profit_amount_raw", "Derived", "final_amount × margin %"),
    ("age_group", "Derived", "Under 18, 18-25, 26-35, 36-45, 46-60, 60+, Unknown"),
    ("discount_pct, discount_band", "Derived", "Discount as % of order value, grouped"),
    ("distance_band, delivery_time_band, delivery_performance", "Derived", "Binned distance/time; Fast/Standard/Slow/Very Slow"),
    ("expected_delivery_min, delivery_delay_min, is_late", "Derived", f"Expected = {BASE_PREP_MIN:.0f} + {MIN_PER_KM} × km; delay = actual − expected"),
    ("order_month, is_cancelled", "Derived", "YYYY-MM; 1 if cancelled"),
    ("*_imputed flags", "Derived", "1 where age / delivery time / distance / order value was filled with the median"),
], columns=["Column(s)", "Type", "Meaning"])

DOC_KPI = """
| KPI | Definition |
|---|---|
| **Total Orders** | All orders in the filter (delivered + cancelled) |
| **Total Revenue** | Σ `final_amount` on **delivered** orders |
| **Average Order Value** | Revenue ÷ delivered orders |
| **Average Delivery Time** | Mean `delivery_time_min`, delivered orders |
| **Cancellation Rate** | Cancelled ÷ total orders |
| **Average Delivery Rating** | Mean `delivery_rating` (delivered orders only) |
| **Profit Margin %** | Σ profit ÷ Σ revenue on delivered orders (order-weighted, not a simple average) |
| **Late %** | Share of delivered orders with `delivery_delay_min > 0` |

Arrows on the KPI cards compare the **last month in the filter with the month before**.
"""

DOC_CAVEATS = """
- **The data looks synthetic / evenly random.** Cities, cuisines, payment modes, ratings and discounts are almost
  perfectly uniform, so differences between categories are small. The dashboard reports what the data
  says, even when the honest answer is "no meaningful difference".
- **Distance and delivery time are essentially uncorrelated** here (see Findings). The mock-up image showed
  r = 0.72, but that number is not in this dataset.
- **Missing data is heavy** (up to 55% of `final_amount`, 50% of age, 33% of delivery time/distance/order value).
  Exact arithmetic reconstructs amounts; remaining numeric gaps use the median and are flagged `*_imputed`;
  distance/delay charts use only rows where both values were really recorded.
- **Restaurant ID and name do not match one-to-one** (900 IDs vs 500 names; each ID has ~100 different names).
  The app treats `restaurant_id` as the entity and shows its most common name.
- **`order_time` is always 00:00**, so there is no hour-of-day analysis; the supplied `Peak_Hour` flag is used.
- **Negative profit margins (28% of orders)** are not necessarily errors: discounts can push an order below
  break-even. The default floors them at 0 as the brief asks; switch to *Raw* in the sidebar to see true margin.
- **Revenue excludes cancelled orders.** Cancelled orders have their delivery rating removed (a cancelled order
  cannot be rated).
"""

DOC_RUN = """
### Run locally (one command)
```bash
pip install -r requirements.txt && streamlit run app.py
```
Or double-click `run_app.bat` (Windows) / run `./run_app.sh` (Mac/Linux): they create a virtual environment,
install the requirements and start the app. First launch builds the database (~15-40 s); later launches are instant.

### Use a different CSV
Replace the file in `data/raw/` (or delete it and upload one in the app). The database rebuilds automatically.

### Publish to GitHub
```bash
git init
git add .
git commit -m "Online Food Delivery Analysis dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/food-delivery-analysis.git
git push -u origin main
```
Create the empty repository first on github.com (**New repository**, no README). The generated `data/processed/` files
are git-ignored; the raw CSV (14.6 MB) is small enough to commit.

### Deploy free on Streamlit Community Cloud
Push to GitHub → share.streamlit.io → **New app** → pick the repo → main file `app.py`.
"""

DOC_RECS = """
Recommendations are written as *what to do given the pattern*. Check the live numbers above before acting.

1. **Cancellations (~15%).** Tackle the biggest recorded reason first, and fix data capture: a large share of
   cancelled orders have no reason at all ('Not Specified'), which blocks root-cause analysis.
2. **Delivery speed.** Because distance barely explains delivery time, look at restaurant preparation time, rider
   allocation and batching rather than only shrinking delivery radius.
3. **Discounts.** Compare profit per order across the discount tiers in *Revenue & Profit*; if higher discounts do
   not lift volume or margin, cap them.
4. **Data quality.** Capture order time, make restaurant ID↔name consistent and reduce missing city/cuisine/payment
   values at source: today they block hour-level demand planning and restaurant benchmarking.
5. **Regional strategy.** Demand is evenly spread across cities; use city-level differences in delivery time and
   cancellations (not volume) to prioritise operational investment.
"""


def page_docs(f: dict) -> None:
    meta = get_meta(CTX["sig"])
    t = st.tabs(["About", "Data dictionary", "Cleaning & features", "SQL design", "KPIs",
                 "Findings & recommendations", "Caveats", "Run & publish"])
    with t[0]:
        st.markdown(DOC_ABOUT)
    with t[1]:
        table(DOC_DICT, height=620)
    with t[2]:
        st.markdown("The cleaning log below is produced by the pipeline itself, so the numbers are exact for "
                    "your file.")
        table(pd.DataFrame(json.loads(meta["cleaning_log"])), height=560)
        st.caption(f"Pipeline v{PIPELINE_VERSION} · database built {meta['built_at']}")
    with t[3]:
        st.markdown("**Design:** a star schema (fact `orders` + 7 dimension tables) with primary keys, foreign keys, "
                    "CHECK constraints and indexes. `vw_orders` joins everything for analysis. "
                    "See the *SQL Explorer* page for the ER diagram, DDL and 16 ready-made queries.")
        st.code("SELECT city, ROUND(SUM(final_amount),0) AS revenue\nFROM vw_orders\n"
                "WHERE order_status = 'Delivered'\nGROUP BY city ORDER BY revenue DESC;", language="sql")
    with t[4]:
        st.markdown(DOC_KPI)
    with t[5]:
        st.markdown("#### Live findings (full dataset, ignoring sidebar filters)")
        for line in live_findings():
            st.markdown(f"- {line}")
        st.markdown("#### Recommendations")
        st.markdown(DOC_RECS)
    with t[6]:
        st.markdown(DOC_CAVEATS)
    with t[7]:
        st.markdown(DOC_RUN)


# ----------------------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------------------
def reset_filters(opts: dict) -> None:
    st.session_state["f_city"] = opts["cities"]
    st.session_state["f_cuisine"] = opts["cuisines"]
    st.session_state["f_status"] = opts["statuses"]
    st.session_state["f_dates"] = (opts["dmin"], opts["dmax"])
    st.session_state["f_basis"] = "Corrected (negatives → 0%)"


def main() -> None:
    st.set_page_config(page_title="FoodDash · Online Food Delivery Analysis", page_icon="🍔", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    _PLOT_N[0] = 0

    raw = find_raw_csv()
    if raw is None:
        st.title("🍔 Online Food Delivery Analysis")
        st.info("No CSV found in `data/raw/`. Upload your dataset to begin.")
        up = st.file_uploader("Upload the food-delivery CSV", type="csv")
        if up is not None:
            (RAW_DIR / "uploaded_dataset.csv").write_bytes(up.getvalue())
            st.rerun()
        return

    sig = file_signature(raw)
    prepare_database(str(raw), sig)
    CTX["sig"] = sig
    meta = get_meta(sig)

    ob = sql_df("SELECT MIN(order_date) AS a, MAX(order_date) AS b FROM vw_orders").iloc[0]
    opts = {
        "cities": sql_df("SELECT DISTINCT city FROM vw_orders ORDER BY city")["city"].tolist(),
        "cuisines": sql_df("SELECT DISTINCT cuisine_type FROM vw_orders ORDER BY cuisine_type")["cuisine_type"].tolist(),
        "statuses": ["Delivered", "Cancelled"],
        "dmin": datetime.strptime(ob["a"], "%Y-%m-%d").date(), "dmax": datetime.strptime(ob["b"], "%Y-%m-%d").date(),
    }
    for key, val in [("f_city", opts["cities"]), ("f_cuisine", opts["cuisines"]), ("f_status", opts["statuses"]),
                     ("f_dates", (opts["dmin"], opts["dmax"])), ("f_basis", "Corrected (negatives → 0%)"),
                     ("page", "Overview")]:
        st.session_state.setdefault(key, val)

    # ------------------------------ sidebar ------------------------------------------
    with st.sidebar:
        st.markdown('<div class="brand">🍔 FoodDash</div><div class="brandsub">Data-Driven Food Delivery Insights</div>',
                    unsafe_allow_html=True)
        for name, (icon, desc) in PAGES.items():
            if st.button(f"{icon}  {name}", key=f"nav_{name}", help=desc,
                         type="primary" if st.session_state["page"] == name else "secondary",
                         **stretch(st.button)):
                st.session_state["page"] = name
                st.rerun()
        st.divider()
        h1, h2 = st.columns([1, 1])
        h1.markdown("**Filters**")
        h2.button("Reset all", on_click=reset_filters, args=(opts,), help="Restore every filter to its default.",
                  key="reset", **stretch(st.button))
        st.multiselect("City", opts["cities"], key="f_city",
                       help="Show only orders from the selected cities. 'Unknown' = city missing in the source.")
        st.multiselect("Cuisine", opts["cuisines"], key="f_cuisine",
                       help="Show only the selected cuisines. 'Unknown' = cuisine missing in the source.")
        st.multiselect("Order status", opts["statuses"], key="f_status",
                       help="Delivered, Cancelled, or both. Revenue always counts delivered orders only.")
        st.date_input("Date range", key="f_dates", min_value=opts["dmin"], max_value=opts["dmax"],
                      help="Pick a start and end date. Orders with an unknown date are only included while the "
                           "full range is selected.")
        st.radio("Profit-margin basis", ["Corrected (negatives → 0%)", "Raw (as supplied)"], key="f_basis",
                 help="Corrected floors negative margins at 0% as the project brief asks. Raw keeps real losses "
                      "so you can see how discounts hurt profit.")
        st.download_button("⬇ Download Cleaned Data", cleaned_csv_bytes(sig), "cleaned_food_delivery_data.csv",
                           "text/csv", help="The full cleaned and feature-engineered dataset as CSV "
                           "(all rows, ignores the filters).", type="primary", **stretch(st.download_button))
        st.caption("Get the processed dataset as CSV")

    # ------------------------------ filters -> dict ----------------------------------
    sel_c, sel_q, sel_s = st.session_state["f_city"], st.session_state["f_cuisine"], st.session_state["f_status"]
    dates = st.session_state["f_dates"]
    f = {
        "cities": None if set(sel_c) == set(opts["cities"]) else tuple(sel_c),
        "cuisines": None if set(sel_q) == set(opts["cuisines"]) else tuple(sel_q),
        "statuses": None if set(sel_s) == set(opts["statuses"]) else tuple(sel_s),
        "date_from": None, "date_to": None,
        "profit_col": "profit_amount" if st.session_state["f_basis"].startswith("Corrected") else "profit_amount_raw",
    }
    if isinstance(dates, (tuple, list)) and len(dates) == 2 and (dates[0], dates[1]) != (opts["dmin"], opts["dmax"]):
        f["date_from"], f["date_to"] = dates[0].isoformat(), dates[1].isoformat()

    # ------------------------------ header -------------------------------------------
    st.markdown(f"""
    <div class="hero"><div><h1>Online Food Delivery Analysis</h1>
    <p>Understand customer behavior, revenue, delivery operations and restaurant performance</p></div>
    <div style="display:flex; gap:.6rem;">
      <div class="badge" title="Rows in the cleaned SQL database">Dataset loaded<b>{int(meta['clean_rows']):,} orders</b></div>
      <div class="badge" title="When the cleaned database was last built">Last updated<b>{meta['built_at']}</b></div>
    </div></div>""", unsafe_allow_html=True)

    page = st.session_state["page"]
    if page in ("SQL Explorer", "Documentation"):
        {"SQL Explorer": page_sql, "Documentation": page_docs}[page](f)
        return
    if not sel_c or not sel_q or not sel_s:
        st.warning("Select at least one value in every filter.")
        return
    if run(SQL["kpi"], f).iloc[0]["orders"] == 0:
        st.warning("No orders match these filters. Widen the filters or press Reset all.")
        return
    {"Overview": page_overview, "Customers & Orders": page_customers, "Revenue & Profit": page_revenue,
     "Delivery Performance": page_delivery, "Restaurants & Cancellations": page_restaurants,
     "Data Explorer": page_data}[page](f)


if __name__ == "__main__":
    main()
