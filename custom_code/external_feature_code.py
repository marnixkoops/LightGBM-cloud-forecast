###########
# PRICING #
###########
# Prepare price data from shards
# datapath = './data/price/' # Combine actuals CSV shards into one file
# files = glob.glob(os.path.join(datapath, '*.csv'))

# np_array_list = []
# for file_ in tqdm(files):
#     df = pd.read_csv(file_, index_col=None, header=0)
#     np_array_list.append(df.as_matrix())    #
# comb_np_array = np.vstack(np_array_list)
# price_df = pd.DataFrame(comb_np_array)
# del files, np_array_list, comb_np_array, df

# price_df.columns = ['product_id', 'date', 'price']
# price_df['date'] = price_df['date'].apply(lambda x: x[0:10])  # keep only date, remove timezone and stuff
# price_df['date'] = pd.to_datetime(price_df['date'])
# price_df['product_id'] = price_df['product_id'].astype('int32')
# price_df['price'] = price_df['price'].astype('float16')
# price_df.reset_index(drop=True, inplace=True)
# price_df.sort_values(by=['date', 'product_id'], inplace=True, ascending=True)

# # Add lagged prices per product
# price_df['price_shift'] = price_df.groupby(['product_id'])['price'].shift(7)

# # Impute now missing first observations per product by backfilling
# price_df['price_shift'] = price_df.groupby(['product_id'])['price_shift'].fillna(method='backfill')
# price_df['date'] = pd.to_datetime(price_df['date'])
# price_df['product_id'] = price_df['product_id'].astype('int32')
# price_df['price'] = price_df['price'].astype('float16')
# price_df['price_shift'] = price_df['price_shift'].astype('float16')
# # Save pricing to disk
# price_df.to_csv('./data/price_df.csv', index=False)

###########
# WEATHER #
###########
# Prepare weather df
# weather_df = pd.read_csv('./data/weather_df.csv')
# weather_df.columns = ['forecast_on', 'forecast_for', 'min_temp_max', 'max_temp_min']
# weather_df['forecast_on'] = pd.to_datetime(weather_df['forecast_on'])
# weather_df['forecast_for'] = pd.to_datetime(weather_df['forecast_for'])

# # Keep only one day ahead forecast by taking last observation of ascendingly ordered duplicate forecast_for dates
# weather_df.drop_duplicates(subset=['forecast_for'], keep='last', inplace=True)
# weather_df['forecast_for'].nunique() == weather_df.shape[0] # Check if we have only unique entries now
# weather_df.drop('forecast_on', axis=1, inplace=True)
# weather_df[['min_temp_max', 'max_temp_min']] = weather_df[['min_temp_max', 'max_temp_min']].astype('int8')
# weather_df.to_hdf('./data/weather_df.h5', 'weather_df')

# # Add prices to dataframe
# print('Merging lagged product pricing into dataframe ...')
# demand_df = pd.merge(demand_df, price_df, on=['product_id', 'date'], how='left')
#
# # Add temperatures to dataframe
# print('Merging lagged weather into dataframe ...')
# demand_df = pd.merge(demand_df, weather_df, on='date', how='left')