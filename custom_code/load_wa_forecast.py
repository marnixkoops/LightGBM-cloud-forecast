import pandas as pd


def load_wa_forecast():
    print('Loading WA forecast')
    # We create the WA from scratch, because we don't have a default_value in the table until March 2017
    # and it seems like there is data missing here and there in the table for the default value
    query = '''
with actuals as (
SELECT
  product_id,
  date,
  on_stock,
  -- We want to start being able to forecast day level outliers again, so our base actuals should include these
  -- Since we introduced this second level of outlier cleaning not too long ago, we want to take actual_intermediate
  -- if it's available and if not, take the actual column (which at that point were the actuals with those outliers)
  coalesce(actual_intermediate, actual) as actual
FROM
  `coolblue-bi-platform-prod.demand_forecast.actual`
where
  date > date('2016-01-01')
),

-- CTE to calculate the WA according to current logic: use the last 21 days that the product was on stock
-- This removes a bunch of dates from the data, so we have to re-add those again later
actuals_lagged as (
SELECT
  date,
  product_id,
  actual,
  sum(actual) over (partition by product_id order by date rows between 1 PRECEDING and 1 PRECEDING) as actual_past_day1,
  sum(actual) over (partition by product_id order by date rows between 2 PRECEDING and 2 PRECEDING) as actual_past_day2,
  sum(actual) over (partition by product_id order by date rows between 3 PRECEDING and 3 PRECEDING) as actual_past_day3,
  sum(actual) over (partition by product_id order by date rows between 4 PRECEDING and 4 PRECEDING) as actual_past_day4,
  sum(actual) over (partition by product_id order by date rows between 5 PRECEDING and 5 PRECEDING) as actual_past_day5,
  sum(actual) over (partition by product_id order by date rows between 6 PRECEDING and 6 PRECEDING) as actual_past_day6,
  sum(actual) over (partition by product_id order by date rows between 7 PRECEDING and 7 PRECEDING) as actual_past_day7,
  sum(actual) over (partition by product_id order by date rows between 8 PRECEDING and 8 PRECEDING) as actual_past_day8,
  sum(actual) over (partition by product_id order by date rows between 9 PRECEDING and 9 PRECEDING) as actual_past_day9,
  sum(actual) over (partition by product_id order by date rows between 10 PRECEDING and 10 PRECEDING) as actual_past_day10,
  sum(actual) over (partition by product_id order by date rows between 11 PRECEDING and 11 PRECEDING) as actual_past_day11,
  sum(actual) over (partition by product_id order by date rows between 12 PRECEDING and 12 PRECEDING) as actual_past_day12,
  sum(actual) over (partition by product_id order by date rows between 13 PRECEDING and 13 PRECEDING) as actual_past_day13,
  sum(actual) over (partition by product_id order by date rows between 14 PRECEDING and 14 PRECEDING) as actual_past_day14,
  sum(actual) over (partition by product_id order by date rows between 15 PRECEDING and 15 PRECEDING) as actual_past_day15,
  sum(actual) over (partition by product_id order by date rows between 16 PRECEDING and 16 PRECEDING) as actual_past_day16,
  sum(actual) over (partition by product_id order by date rows between 17 PRECEDING and 17 PRECEDING) as actual_past_day17,
  sum(actual) over (partition by product_id order by date rows between 18 PRECEDING and 18 PRECEDING) as actual_past_day18,
  sum(actual) over (partition by product_id order by date rows between 19 PRECEDING and 19 PRECEDING) as actual_past_day19,
  sum(actual) over (partition by product_id order by date rows between 20 PRECEDING and 20 PRECEDING) as actual_past_day20,
  sum(actual) over (partition by product_id order by date rows between 21 PRECEDING and 21 PRECEDING) as actual_past_day21
FROM
  actuals
WHERE
  on_stock = True
),

forecast as (
select
  date,
  product_id,
  actual,
  (0.7 * actual_past_day1 + 0.6666666666666666 * actual_past_day2 + 0.6333333333333333 * actual_past_day3 +
  0.6 * actual_past_day4 + 0.5666666666666667 * actual_past_day5 + 0.5333333333333333 * actual_past_day6 +
  0.49999999999999994 * actual_past_day7 + 0.35000000000000003 * actual_past_day8 + 0.3333333333333333 * actual_past_day9 +
  0.31666666666666665 * actual_past_day10 + 0.3 * actual_past_day11 + 0.2833333333333333 * actual_past_day12 +
  0.26666666666666666 * actual_past_day13 + 0.24999999999999997 * actual_past_day14 + 0.11666666666666668 * actual_past_day15 +
  0.11111111111111112 * actual_past_day16 + 0.10555555555555557 * actual_past_day17 + 0.1 * actual_past_day18 +
  0.09444444444444444 * actual_past_day19 + 0.08888888888888889 * actual_past_day20 + 0.08333333333333333 * actual_past_day21) / 7 as wa
from
  actuals_lagged a
)

select
  a.product_id,
  a.date,
  -- This clause uses the last known WA value when the current WA value is NULL (due to out of stock)
  FIRST_VALUE(wa IGNORE NULLS) OVER (PARTITION BY a.product_id ORDER BY a.date DESC ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) as wa
from
  actuals a
  left join forecast f on f.date = a.date and f.product_id = a.product_id
WHERE
  EXISTS (
    SELECT
      'x'
    FROM
      `master.product` p
    WHERE
      p.product_id = a.product_id
      -- PROMO PTs provided by Arthur
      -- and p.product_type_id in (2452, 2458, 2096, 2090, 2048, 2250, 2562, 9504)
      -- and a.product_id in (785998)
  )
'''

    return pd.read_gbq(query, 'coolblue-bi-platform-prod', dialect='standard')
