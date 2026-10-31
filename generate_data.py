"""
Generates a reproducible SYNTHETIC food-delivery orders dataset (data/orders.csv).

The data is simulated (seeded), not scraped from any real company. Relationships
(traffic, weather, distance, prep time -> delivery time -> ratings/cancellations)
are built in so the analysis has realistic patterns to discover.

Run:  python generate_data.py
"""
import numpy as np
import pandas as pd

SEED = 42
N = 30_000
rng = np.random.default_rng(SEED)

cities = {"Pune": 0.20, "Mumbai": 0.24, "Bengaluru": 0.22, "Delhi": 0.20, "Hyderabad": 0.14}
city_speed = {"Pune": 1.00, "Mumbai": 1.18, "Bengaluru": 1.15, "Delhi": 1.10, "Hyderabad": 1.02}
cuisines = {"North Indian": 0.26, "Biryani": 0.18, "Fast Food": 0.17, "Chinese": 0.12,
            "South Indian": 0.11, "Pizza": 0.08, "Desserts": 0.05, "Healthy": 0.03}
cuisine_prep = {"North Indian": 17, "Biryani": 19, "Fast Food": 11, "Chinese": 14,
                "South Indian": 12, "Pizza": 18, "Desserts": 8, "Healthy": 13}
cuisine_value = {"North Indian": 380, "Biryani": 420, "Fast Food": 260, "Chinese": 340,
                 "South Indian": 230, "Pizza": 450, "Desserts": 210, "Healthy": 400}
weather_p = {"Clear": 0.62, "Cloudy": 0.18, "Rain": 0.15, "Fog": 0.05}
weather_f = {"Clear": 1.00, "Cloudy": 1.03, "Rain": 1.22, "Fog": 1.12}
traffic_f = {"Low": 0.85, "Medium": 1.00, "High": 1.25, "Jam": 1.55}
payments = {"UPI": 0.52, "Card": 0.20, "Cash": 0.15, "Wallet": 0.13}

dates = pd.date_range("2026-01-01", "2026-09-30", freq="D")
# mild weekend uplift
day_w = np.where(dates.dayofweek >= 5, 1.25, 1.0) * np.linspace(0.9, 1.15, len(dates))
order_date = rng.choice(dates, size=N, p=day_w / day_w.sum())

# lunch and dinner peaks
hour_w = np.array([1, 0.5, 0.3, 0.2, 0.2, 0.4, 1, 2, 3, 3, 4, 7, 10, 9, 6, 4, 4, 5, 7, 11, 13, 12, 8, 3], float)
order_hour = rng.choice(24, size=N, p=hour_w / hour_w.sum())

def pick(d, n):
    return rng.choice(list(d.keys()), size=n, p=np.array(list(d.values())) / sum(d.values()))

city = pick(cities, N)
cuisine = pick(cuisines, N)
weather = pick(weather_p, N)
payment = pick(payments, N)

# traffic depends on the hour (peak hours are worse)
peak = np.isin(order_hour, [12, 13, 19, 20, 21])
traffic = np.empty(N, dtype=object)
for i, is_peak in enumerate([False, True]):
    mask = peak == is_peak
    probs = [0.20, 0.40, 0.28, 0.12] if is_peak else [0.45, 0.35, 0.15, 0.05]
    traffic[mask] = rng.choice(["Low", "Medium", "High", "Jam"], size=mask.sum(), p=probs)

distance = np.clip(rng.gamma(shape=3.0, scale=1.6, size=N), 0.5, 15).round(1)
n_items = rng.integers(1, 7, size=N)
base_val = np.array([cuisine_value[c] for c in cuisine]) * (0.55 + 0.22 * n_items) * rng.lognormal(0, 0.18, N)
discount = rng.choice([0, 5, 10, 15, 20, 30], size=N, p=[0.38, 0.12, 0.18, 0.14, 0.12, 0.06])
order_value = (base_val * (1 - discount / 100)).round(0)

prep = np.array([cuisine_prep[c] for c in cuisine]) * rng.lognormal(0, 0.22, N) * np.where(peak, 1.18, 1.0)
prep = prep.clip(4, 60).round(1)

speed = np.array([city_speed[c] for c in city])
tf = np.array([traffic_f[t] for t in traffic])
wf = np.array([weather_f[w] for w in weather])
travel = distance * 3.4 * speed * tf * wf * rng.lognormal(0, 0.12, N)  # ~3.4 min per km baseline
pickup_wait = rng.exponential(2.5, N) + np.where(peak, 2.0, 0)
delivery_time = (prep + pickup_wait + travel + 3).clip(10, 120).round(1)

promised = (np.ceil((12 + distance * 3.6 + np.array([cuisine_prep[c] for c in cuisine])) / 5) * 5 + 5).astype(int)
is_late = (delivery_time > promised).astype(int)

# cancellations rise with long waits and with cash/high discount noise
cancel_p = 0.012 + 0.0016 * np.clip(delivery_time - 35, 0, None) + 0.01 * (payment == "Cash")
cancelled = (rng.random(N) < cancel_p.clip(0, 0.4)).astype(int)

rider_id = rng.integers(1, 601, size=N)
rider_skill = rng.normal(0, 1, 601)
delivery_time = (delivery_time - 1.2 * rider_skill[rider_id - 1]).clip(10, 120).round(1)
is_late = (delivery_time > promised).astype(int)

# customer rating falls with lateness
cust = 4.7 - 0.045 * np.clip(delivery_time - promised, 0, None) - 0.15 * (is_late) + rng.normal(0, 0.45, N)
customer_rating = np.clip(np.round(cust * 2) / 2, 1, 5)
customer_rating = np.where(cancelled == 1, np.nan, customer_rating)

df = pd.DataFrame({
    "order_id": [f"ORD{100000 + i}" for i in range(N)],
    "order_date": pd.to_datetime(order_date),
    "order_hour": order_hour,
    "city": city, "cuisine": cuisine, "payment_mode": payment,
    "weather": weather, "traffic": traffic, "rider_id": rider_id,
    "distance_km": distance, "n_items": n_items, "discount_pct": discount,
    "order_value": order_value, "prep_time_min": prep,
    "promised_time_min": promised, "delivery_time_min": delivery_time,
    "is_late": is_late, "cancelled": cancelled, "customer_rating": customer_rating,
}).sort_values(["order_date", "order_hour"]).reset_index(drop=True)

import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/orders.csv", index=False)
print(f"Saved data/orders.csv  rows={len(df):,}  cols={df.shape[1]}")
