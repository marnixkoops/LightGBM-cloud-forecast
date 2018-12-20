import pandas as pd


def load_product():
    print('Loading product')
    query = '''
SELECT
  product_id,
  product_type_id,
  brand_id,
  manufacturer_id,
  product_group_id,
  team_id,
  subproduct_type_id
FROM
  `coolblue-bi-platform-prod.master.product`
'''

    return pd.read_gbq(query, 'coolblue-bi-platform-prod', dialect='standard')
