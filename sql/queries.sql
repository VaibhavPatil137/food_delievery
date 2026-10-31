-- name: Late-delivery rate by traffic and weather
-- desc: Which operating conditions hurt on-time performance the most?
SELECT traffic,
       weather,
       COUNT(*)                                   AS orders,
       ROUND(AVG(delivery_time_min), 1)           AS avg_delivery_min,
       ROUND(100.0 * AVG(is_late), 1)             AS late_pct
FROM orders
GROUP BY traffic, weather
HAVING COUNT(*) >= 100
ORDER BY late_pct DESC;

-- name: City revenue ranking with share of total
-- desc: Window functions (RANK, SUM OVER) to rank cities and show revenue share.
SELECT city,
       ROUND(SUM(order_value), 0)                                    AS revenue,
       RANK() OVER (ORDER BY SUM(order_value) DESC)                  AS revenue_rank,
       ROUND(100.0 * SUM(order_value) / SUM(SUM(order_value)) OVER (), 1) AS revenue_share_pct,
       ROUND(AVG(order_value), 0)                                    AS aov
FROM orders
WHERE cancelled = 0
GROUP BY city
ORDER BY revenue_rank;

-- name: Month-over-month revenue growth
-- desc: CTE + LAG to compute MoM growth.
WITH monthly AS (
    SELECT strftime('%Y-%m', order_date) AS month,
           SUM(order_value)              AS revenue
    FROM orders
    WHERE cancelled = 0
    GROUP BY month
)
SELECT month,
       ROUND(revenue, 0) AS revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
             / LAG(revenue) OVER (ORDER BY month), 1) AS mom_growth_pct
FROM monthly
ORDER BY month;

-- name: Peak-hour demand and late rate
-- desc: Hourly order volume, share of daily demand and on-time performance.
SELECT order_hour,
       COUNT(*)                                        AS orders,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS share_of_orders_pct,
       ROUND(100.0 * AVG(is_late), 1)                  AS late_pct
FROM orders
GROUP BY order_hour
ORDER BY order_hour;

-- name: Top 10 riders by on-time rate (min 40 orders)
-- desc: HAVING to filter out riders with too few orders to judge fairly.
SELECT rider_id,
       COUNT(*)                                  AS orders,
       ROUND(100.0 * (1 - AVG(is_late)), 1)      AS on_time_pct,
       ROUND(AVG(delivery_time_min), 1)          AS avg_delivery_min
FROM orders
GROUP BY rider_id
HAVING COUNT(*) >= 40
ORDER BY on_time_pct DESC, orders DESC
LIMIT 10;

-- name: Cancellation rate by payment mode
-- desc: Does cash-on-delivery drive more cancellations?
SELECT payment_mode,
       COUNT(*)                              AS orders,
       SUM(cancelled)                        AS cancelled_orders,
       ROUND(100.0 * AVG(cancelled), 2)      AS cancel_pct
FROM orders
GROUP BY payment_mode
ORDER BY cancel_pct DESC;

-- name: Delivery-time bucket vs customer rating
-- desc: CASE bucketing to show how waiting time erodes customer satisfaction.
SELECT CASE
         WHEN delivery_time_min < 30 THEN '1. Under 30 min'
         WHEN delivery_time_min < 45 THEN '2. 30-45 min'
         WHEN delivery_time_min < 60 THEN '3. 45-60 min'
         ELSE '4. 60+ min'
       END                               AS delivery_bucket,
       COUNT(*)                          AS rated_orders,
       ROUND(AVG(customer_rating), 2)    AS avg_rating
FROM orders
WHERE cancelled = 0
GROUP BY delivery_bucket
ORDER BY delivery_bucket;
