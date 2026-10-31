"""Food Delivery Operations Analytics - Streamlit dashboard."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import utils

st.set_page_config(page_title="Food Delivery Analytics", page_icon="🛵", layout="wide")

NAVY, BLUE, ORANGE, GREY = "#1F3A5F", "#3B82C4", "#F28C28", "#9AA5B1"
TRAFFIC_ORDER = ["Low", "Medium", "High", "Jam"]
WEATHER_ORDER = ["Clear", "Cloudy", "Fog", "Rain"]
sns.set_theme(style="whitegrid", rc={"axes.spines.top": False, "axes.spines.right": False})


# ------------------------------------------------------------------ caching
@st.cache_data
def get_data() -> pd.DataFrame:
    return utils.load_data()


@st.cache_resource
def get_model(df: pd.DataFrame):
    return utils.train_model(df)


def show(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def bar(data, x, y, title, color=BLUE, fmt="{:.1f}", order=None, xlabel="", ylabel="", horizontal=False):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    if order is not None:
        data = data.set_index(x).loc[[o for o in order if o in set(data[x])]].reset_index()
    if horizontal:
        ax.barh(data[x].astype(str), data[y], color=color)
        ax.invert_yaxis()
        for i, v in enumerate(data[y]):
            ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8)
        ax.set_xlim(0, data[y].max() * 1.18)
    else:
        ax.bar(data[x].astype(str), data[y], color=color)
        for i, v in enumerate(data[y]):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, data[y].max() * 1.18)
        plt.setp(ax.get_xticklabels(), rotation=25 if len(data) > 5 else 0, ha="right" if len(data) > 5 else "center")
    ax.set_title(title, fontsize=11, fontweight="bold", color=NAVY, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- data
df_all = get_data()

st.title("🛵 Food Delivery Operations Analytics")
st.caption("End-to-end analytics project: KPI dashboard, delivery-performance diagnostics, "
           "SQL insights and a delivery-time prediction model. "
           "**Data is synthetic** (seeded simulation, see `generate_data.py`), not from a real company.")

# ------------------------------------------------------------------ filters
st.sidebar.header("Filters")
dmin, dmax = df_all["order_date"].min().date(), df_all["order_date"].max().date()
date_sel = st.sidebar.date_input("Order date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(date_sel, (tuple, list)):
    d_start = date_sel[0]
    d_end = date_sel[1] if len(date_sel) > 1 else date_sel[0]
else:
    d_start = d_end = date_sel
cities = st.sidebar.multiselect("City", sorted(df_all["city"].unique()), default=sorted(df_all["city"].unique()))
cuisines = st.sidebar.multiselect("Cuisine", sorted(df_all["cuisine"].unique()), default=sorted(df_all["cuisine"].unique()))

df = df_all[
    (df_all["order_date"].dt.date >= d_start)
    & (df_all["order_date"].dt.date <= d_end)
    & (df_all["city"].isin(cities))
    & (df_all["cuisine"].isin(cuisines))
]
st.sidebar.markdown(f"**{len(df):,}** of {len(df_all):,} orders selected")

if df.empty:
    st.warning("No orders match the current filters. Widen the date range or select more cities/cuisines.")
    st.stop()

tab_over, tab_perf, tab_rev, tab_sql, tab_ml = st.tabs(
    ["📊 Overview", "⏱️ Delivery performance", "💰 Revenue & customers", "🗃️ SQL explorer", "🤖 Delivery-time predictor"]
)

# ----------------------------------------------------------------- overview
with tab_over:
    k = utils.compute_kpis(df)
    c = st.columns(6)
    c[0].metric("Orders", f"{k['orders']:,}")
    c[1].metric("Revenue (delivered)", f"₹{k['revenue'] / 1e6:.2f}M")
    c[2].metric("Avg order value", f"₹{k['aov']:.0f}")
    c[3].metric("Avg delivery time", f"{k['avg_delivery']:.1f} min")
    c[4].metric("Late deliveries", f"{k['late_pct']:.1f}%")
    c[5].metric("Cancellation rate", f"{k['cancel_pct']:.1f}%")

    monthly = (df.groupby("month")
               .agg(orders=("order_id", "count"),
                    revenue=("order_value", lambda s: s[df.loc[s.index, "cancelled"] == 0].sum()),
                    late_pct=("is_late", lambda s: 100 * s.mean()))
               .reset_index())
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.bar(monthly["month"], monthly["revenue"] / 1e6, color=BLUE)
        ax.set_title("Monthly revenue (₹ million)", fontsize=11, fontweight="bold", color=NAVY, loc="left")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        show(fig)
    with right:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.plot(monthly["month"], monthly["late_pct"], marker="o", color=ORANGE)
        ax.set_title("Late-delivery rate by month (%)", fontsize=11, fontweight="bold", color=NAVY, loc="left")
        ax.set_ylim(0, max(monthly["late_pct"].max() * 1.3, 1))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        show(fig)

    st.subheader("Key findings (update with filters)")
    by_traffic = df.groupby("traffic")["is_late"].mean().mul(100)
    by_weather = df.groupby("weather")["is_late"].mean().mul(100)
    by_city = df.groupby("city")["is_late"].mean().mul(100)
    pay = df.groupby("payment_mode")["cancelled"].mean().mul(100)
    peak = df.groupby("order_hour").size().sort_values(ascending=False).head(3).index.sort_values().tolist()
    findings = [
        f"**Traffic is the biggest driver of lateness:** `{by_traffic.idxmax()}` traffic runs at "
        f"**{by_traffic.max():.1f}%** late vs **{by_traffic.min():.1f}%** for `{by_traffic.idxmin()}`.",
        f"**Weather matters:** `{by_weather.idxmax()}` orders are late {by_weather.max():.1f}% of the time "
        f"vs {by_weather.min():.1f}% in `{by_weather.idxmin()}` conditions.",
        f"**Worst city for punctuality:** `{by_city.idxmax()}` ({by_city.max():.1f}% late); "
        f"best is `{by_city.idxmin()}` ({by_city.min():.1f}%).",
        f"**Cancellations:** `{pay.idxmax()}` payments cancel most ({pay.max():.1f}%) vs {pay.min():.1f}% for `{pay.idxmin()}`.",
        f"**Demand peaks** around hours {', '.join(f'{h}:00' for h in peak)}, where staffing and rider supply matter most.",
    ]
    for f in findings:
        st.markdown("- " + f)

# ------------------------------------------------------ delivery performance
with tab_perf:
    a, b = st.columns(2)
    with a:
        t = df.groupby("traffic")["is_late"].mean().mul(100).reset_index()
        show(bar(t, "traffic", "is_late", "Late-delivery rate by traffic (%)", color=ORANGE, order=TRAFFIC_ORDER, fmt="{:.1f}%"))
    with b:
        w = df.groupby("weather")["is_late"].mean().mul(100).reset_index()
        show(bar(w, "weather", "is_late", "Late-delivery rate by weather (%)", color=ORANGE, order=WEATHER_ORDER, fmt="{:.1f}%"))

    a, b = st.columns(2)
    with a:
        heat = (df.pivot_table(index="traffic", columns="weather", values="is_late", aggfunc="mean") * 100)
        heat = heat.reindex(index=[x for x in TRAFFIC_ORDER if x in heat.index],
                            columns=[x for x in WEATHER_ORDER if x in heat.columns])
        fig, ax = plt.subplots(figsize=(6, 3.4))
        sns.heatmap(heat, annot=True, fmt=".0f", cmap="OrRd", cbar_kws={"label": "% late"}, ax=ax)
        ax.set_title("Late % : traffic × weather", fontsize=11, fontweight="bold", color=NAVY, loc="left")
        ax.set_xlabel("")
        ax.set_ylabel("")
        fig.tight_layout()
        show(fig)
    with b:
        hourly = df.groupby("order_hour").agg(orders=("order_id", "count"), late=("is_late", "mean")).reset_index()
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.bar(hourly["order_hour"], hourly["orders"], color=BLUE, alpha=0.85)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Orders")
        ax2 = ax.twinx()
        ax2.plot(hourly["order_hour"], hourly["late"] * 100, color=ORANGE, marker="o", markersize=3)
        ax2.set_ylabel("Late %")
        ax2.grid(False)
        ax.set_title("Hourly demand and late %", fontsize=11, fontweight="bold", color=NAVY, loc="left")
        fig.tight_layout()
        show(fig)

    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.hist(df["delivery_time_min"], bins=40, color=BLUE, alpha=0.9)
        ax.axvline(df["delivery_time_min"].mean(), color=ORANGE, linestyle="--", label="Mean")
        ax.axvline(df["promised_time_min"].mean(), color=NAVY, linestyle=":", label="Avg promised")
        ax.set_title("Delivery-time distribution (min)", fontsize=11, fontweight="bold", color=NAVY, loc="left")
        ax.legend()
        fig.tight_layout()
        show(fig)
    with b:
        sample = df.sample(min(len(df), 3000), random_state=1)
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.scatter(sample["distance_km"], sample["delivery_time_min"], s=6, alpha=0.35, color=BLUE)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Delivery time (min)")
        ax.set_title("Distance vs delivery time", fontsize=11, fontweight="bold", color=NAVY, loc="left")
        fig.tight_layout()
        show(fig)

    st.subheader("Rider performance")
    riders = (df.groupby("rider_id")
              .agg(orders=("order_id", "count"), on_time_pct=("is_late", lambda s: 100 * (1 - s.mean())),
                   avg_delivery_min=("delivery_time_min", "mean"))
              .query("orders >= 20").sort_values("on_time_pct").round(1))
    if riders.empty:
        st.info("Not enough orders per rider in the current selection (need 20+).")
    else:
        l, r = st.columns(2)
        l.markdown("**Bottom 10 riders by on-time %** (min 20 orders)")
        l.dataframe(riders.head(10))
        r.markdown("**Top 10 riders by on-time %**")
        r.dataframe(riders.tail(10).iloc[::-1])

# ------------------------------------------------------- revenue & customers
with tab_rev:
    deliv = df[df["cancelled"] == 0]
    a, b = st.columns(2)
    with a:
        city_rev = deliv.groupby("city")["order_value"].sum().div(1e6).reset_index().sort_values("order_value", ascending=False)
        show(bar(city_rev, "city", "order_value", "Revenue by city (₹ million)", fmt="{:.2f}"))
    with b:
        cu = deliv.groupby("cuisine")["order_value"].sum().div(1e6).reset_index().sort_values("order_value", ascending=False)
        show(bar(cu, "cuisine", "order_value", "Revenue by cuisine (₹ million)", color=NAVY, fmt="{:.2f}", horizontal=True))

    a, b = st.columns(2)
    with a:
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow = df.groupby("weekday").size().reset_index(name="orders")
        show(bar(dow, "weekday", "orders", "Orders by weekday", order=dow_order, fmt="{:.0f}"))
    with b:
        pay = df.groupby("payment_mode")["cancelled"].mean().mul(100).reset_index().sort_values("cancelled", ascending=False)
        show(bar(pay, "payment_mode", "cancelled", "Cancellation rate by payment mode (%)", color=ORANGE, fmt="{:.2f}%"))

    a, b = st.columns(2)
    with a:
        bins = [0, 30, 45, 60, 200]
        labels = ["<30", "30-45", "45-60", "60+"]
        tmp = deliv.assign(bucket=pd.cut(deliv["delivery_time_min"], bins=bins, labels=labels, right=False))
        rb = tmp.groupby("bucket", observed=True)["customer_rating"].mean().reset_index()
        rb["bucket"] = rb["bucket"].astype(str)
        fig = bar(rb, "bucket", "customer_rating", "Avg customer rating by delivery time (min)", color=BLUE, fmt="{:.2f}")
        fig.axes[0].set_ylim(3.5, 5.0)
        show(fig)
    with b:
        dsc = deliv.groupby("discount_pct").agg(aov=("order_value", "mean"), orders=("order_id", "count")).reset_index()
        show(bar(dsc, "discount_pct", "aov", "Avg order value by discount % (₹)", color=NAVY, fmt="{:.0f}"))
    st.caption("Heavier discounts do not automatically raise basket size here, so promos should be tested "
               "before scaling (a classic A/B-testing follow-up).")

# ------------------------------------------------------------- SQL explorer
with tab_sql:
    st.markdown("Queries run on an in-memory **SQLite** table called `orders` built from your *current filters*.")
    queries = utils.load_queries()
    name = st.selectbox("Choose an analysis query", list(queries.keys()))
    st.caption(queries[name]["desc"])
    st.code(queries[name]["sql"], language="sql")
    try:
        st.dataframe(utils.run_sql(df, queries[name]["sql"]), hide_index=True)
    except Exception as e:  # keep the app alive if a filter leaves too little data
        st.error(f"Query failed: {e}")

    st.divider()
    st.markdown("**Write your own query**")
    custom = st.text_area("SQL", value="SELECT city, COUNT(*) AS orders, ROUND(AVG(delivery_time_min),1) AS avg_min\nFROM orders\nGROUP BY city\nORDER BY orders DESC;", height=120)
    if st.button("Run query"):
        if not custom.strip().lower().startswith(("select", "with")):
            st.error("Only SELECT / WITH queries are allowed.")
        else:
            try:
                st.dataframe(utils.run_sql(df, custom), hide_index=True)
            except Exception as e:
                st.error(f"Query failed: {e}")

# ---------------------------------------------------------------- predictor
with tab_ml:
    st.markdown("A gradient-boosting model trained on all orders predicts delivery time and flags orders at risk of being late.")
    model, m = get_model(df_all)
    c = st.columns(4)
    c[0].metric("Model MAE", f"{m['mae']:.2f} min")
    c[1].metric("Baseline MAE (mean)", f"{m['baseline_mae']:.2f} min")
    c[2].metric("R² (hold-out)", f"{m['r2']:.3f}")
    c[3].metric("Test orders", f"{m['n_test']:,}")
    st.divider()

    l, r = st.columns(2)
    with l:
        city = st.selectbox("City", sorted(df_all["city"].unique()), index=0)
        cuisine = st.selectbox("Cuisine", sorted(df_all["cuisine"].unique()), index=0)
        traffic = st.select_slider("Traffic", options=TRAFFIC_ORDER, value="Medium")
        weather = st.selectbox("Weather", WEATHER_ORDER)
    with r:
        distance = st.slider("Distance (km)", 0.5, 15.0, 4.5, 0.5)
        prep = st.slider("Restaurant prep time (min)", 4, 45, int(utils.CUISINE_PREP[cuisine]))
        hour = st.slider("Order hour", 0, 23, 20)
        items = st.slider("Items in order", 1, 6, 3)

    pred = utils.predict_one(model, traffic=traffic, weather=weather, city=city, cuisine=cuisine,
                             distance_km=distance, prep_time_min=prep, order_hour=hour, n_items=items)
    promised = utils.promised_time(distance, cuisine)
    a, b, c3 = st.columns(3)
    a.metric("Predicted delivery time", f"{pred:.0f} min")
    b.metric("Promised time", f"{promised} min")
    c3.metric("Buffer", f"{promised - pred:+.0f} min")
    if pred > promised:
        st.error("⚠️ High risk of a late delivery. Consider assigning a faster rider or extending the promised time.")
    elif promised - pred < 5:
        st.warning("Tight margin: small delays could make this order late.")
    else:
        st.success("✅ Comfortable buffer: this order is likely to arrive on time.")
