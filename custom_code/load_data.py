import pandas as pd


def load_data():
    print('Loading data')
    query = '''
SELECT
  product_id,
  date,
  actual,
  on_stock
FROM (
  SELECT
    product_id,
    date,
    -- We want to start being able to forecast day level outliers again, so our base actuals should include these
    -- Since we introduced this second level of outlier cleaning not too long ago, we want to take actual_intermediate
    -- if it's available and if not, take the actual column (which at that point were the actuals with those outliers)
    coalesce(a.actual_intermediate, a.actual) as actual,
    on_stock,
    COUNT(*) OVER (PARTITION BY a.product_id) AS nr_days
  FROM
    `coolblue-bi-platform-prod.demand_forecast.actual` a
  WHERE
    date > DATE('2016-01-01') 
) a
WHERE
  EXISTS (
    SELECT 
      'x' 
    FROM 
      `master.product` p 
    WHERE
      p.product_id = a.product_id
      -- PTs provided by Arthur
      -- and p.product_type_id in (2369)-- (4396, 4999, 5539, 2369, 2703, 2233, 2341, 2627, 5600, 2063)
      -- and p.product_type_id in (2452, 2458, 2096, 2090, 2048, 2250, 2562, 9504)
      -- and a.product_id in (785998)
  )
  -- Epic focuses on days with at least one year of data for seasonality patterns
  AND nr_days >= 365
'''

    return pd.read_gbq(query, 'coolblue-bi-platform-prod', dialect='standard')
