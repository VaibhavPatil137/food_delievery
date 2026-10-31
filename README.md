# 🛵 Food Delivery Operations Analytics

An end-to-end data analytics project: KPI dashboard, delivery-performance diagnostics, SQL insights and a
delivery-time prediction model, packaged as an interactive **Streamlit** app.

**Live app:** `https://<your-app-name>.streamlit.app`  <!-- replace after deploying -->

> **Note on data:** the dataset is **synthetic**. `generate_data.py` simulates 30,000 orders
> (Jan-Sep 2026, 5 Indian cities) with a fixed random seed, so results are fully reproducible.
> It is not real company data.

## Business questions answered
1. What drives late deliveries: traffic, weather, city, time of day, or rider?
2. Where does revenue come from (city, cuisine, weekday) and how do discounts affect basket size?
3. Which payment modes lead to more cancellations?
4. How does waiting time affect customer ratings?
5. Can we predict delivery time and flag orders at risk of being late?

## Key findings (full 30,000-order dataset)
- **Traffic is the #1 lateness driver:** 50.2% of orders in `Jam` traffic are late vs 2.4% in `Low` traffic.
- **Rain makes it worse:** 22.9% late in rain vs 11.2% in clear weather; `Jam + Rain` orders are late **66%** of the time.
- **City gap:** Mumbai is the least punctual (17.1% late), Pune the most (8.8%).
- **Cash-on-delivery cancels most:** 3.94% vs 2.58% for card payments.
- **Long waits hurt ratings:** average rating drops to ~4.3 for 60+ minute deliveries vs ~4.6 under 45 minutes.
- **Model:** gradient boosting predicts delivery time with **MAE 2.9 min** (vs 11.1 min for a mean baseline) and **R² 0.93** on a 20% hold-out set.

## What's inside
| Tab | Contents |
|---|---|
| Overview | KPIs, monthly revenue, late-rate trend, auto-generated key findings |
| Delivery performance | Late % by traffic/weather, traffic x weather heatmap, hourly demand, rider leaderboard |
| Revenue & customers | Revenue by city/cuisine, weekday demand, cancellations, rating vs wait time, discounts |
| SQL explorer | 7 analysis queries (CTEs, window functions, CASE, HAVING) on SQLite + a custom query box |
| Delivery-time predictor | Interactive prediction with on-time risk flag |

## Tech stack
Python, Pandas, NumPy, SQL (SQLite), scikit-learn (HistGradientBoostingRegressor), Matplotlib, Seaborn, Streamlit.

## Run locally
```bash
git clone https://github.com/<your-username>/food-delivery-analytics.git
cd food-delivery-analytics
pip install -r requirements.txt
python generate_data.py      # optional: regenerates data/orders.csv (already included)
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (free)
1. Push this repo to GitHub (public).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose this repo, branch `main`, main file `app.py`, then **Deploy**.
4. Copy the generated `*.streamlit.app` URL into this README and your resume.

## Project structure
```
food-delivery-analytics/
├── app.py              # Streamlit dashboard
├── utils.py            # data loading, KPIs, SQL engine, ML model
├── generate_data.py    # reproducible synthetic dataset generator
├── sql/queries.sql     # analysis queries shown in the SQL explorer
├── data/orders.csv     # 30,000 simulated orders
├── requirements.txt
└── README.md
```

## Dataset columns
`order_id, order_date, order_hour, city, cuisine, payment_mode, weather, traffic, rider_id, distance_km,
n_items, discount_pct, order_value, prep_time_min, promised_time_min, delivery_time_min, is_late,
cancelled, customer_rating`

## Possible extensions
- Replace the synthetic data with a real delivery dataset (e.g. from Kaggle) and compare findings.
- Add an A/B-test module for discount campaigns.
- Add late-delivery classification and SHAP explanations.
