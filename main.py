from elia_data import get_elia_data_ods001
from wx_data_openmeteo import get_wx_df
from functions import featurize_datetime_index
import os
import pandas as pd
import numpy as np
import holidays
import datetime
import matplotlib.pyplot as plt


# Read data, remove nulls and redundant columns
data = get_elia_data_ods001()
data = data.dropna()
data.drop('Resolution code', axis=1, inplace=True)

# standardise to utc due to daylight saving & set date as index
data['DateTime'] = pd.to_datetime(data['DateTime'], utc=True)
data.set_index('DateTime', inplace=True)
data = data.sort_index()

# weather data
start_date = data.index.min().strftime('%Y-%m-%d') # first date in load data
end_date = data.index.max().strftime('%Y-%m-%d') # last date in load data
freq = 'hourly'
wx_params = ['temperature_2m', 'precipitation', 'apparent_temperature']

df_wx_antwerp = get_wx_df(start_date, end_date, freq, wx_params, request_type='historical', city='Antwerp')
df_wx_ghent = get_wx_df(start_date, end_date, freq, wx_params, request_type='historical', city='Ghent')
df_wx_charleroi = get_wx_df(start_date, end_date, freq, wx_params, request_type='historical', city='Charleroi')

df_wx_merged = df_wx_antwerp.merge(df_wx_ghent, 
                how = 'left', 
                left_index = True,
                right_index = True)

df_wx_merged = df_wx_merged.merge(df_wx_charleroi, 
                how = 'left', 
                left_index = True,
                right_index = True)

# limit data frame till the day before the model run date
today = datetime.date.today()

one_day_back = (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
start_date = '2015-01-01'

df = data[start_date:one_day_back]`# both included since date time index`


# add public holiday data
country_code = 'BE'
be_holidays = holidays.country_holidays(country_code)
df_hols = df[df.index.map(lambda x: x in be_holidays)]
df['is_holiday'] = np.where(df.index.isin(df_hols.index), 1, 0)

# add calendar features to dataset
df = featurize_datetime_index(df)

# resample weather df to match that of the elec load
df_wx_merged = df_wx_merged.resample('15T').mean().ffill()

# join features together in 1 dataframe
df_features_merged = df.merge(df_wx_merged, 
                how = 'left', 
                left_index = True,
                right_index = True)


# add lagged features
df_features_merged['TotalLoadMaxLagged72h'] = df_features_merged['Total Load'].sort_index().rolling(72*4).max()
df_features_merged['TotalLoadMinLagged72h'] = df_features_merged['Total Load'].sort_index().rolling(72*4).min()
df_features_merged['TotalLoadMeanLagged72h'] = df_features_merged['Total Load'].sort_index().rolling(72*4).mean()
df_features_merged['TotalLoadMaxLagged96h'] = df_features_merged['Total Load'].sort_index().rolling(96*4).max()
df_features_merged['TotalLoadMinLagged96h'] = df_features_merged['Total Load'].sort_index().rolling(96*4).min()
df_features_merged['TotalLoadMeanLagged96h'] = df_features_merged['Total Load'].sort_index().rolling(96*4).mean()
df_features_merged['TotalLoadMinTrend96to72'] =  (df_features_merged['TotalLoadMinLagged96h'] - df_features_merged['TotalLoadMinLagged72h'])/df_features_merged['TotalLoadMinLagged96h']
df_features_merged['TotalLoadMaxTrend96to72'] =  (df_features_merged['TotalLoadMaxLagged96h'] - df_features_merged['TotalLoadMaxLagged72h'])/df_features_merged['TotalLoadMaxLagged96h']


# format features
df_features_merged.dropna()
df_features_merged['week_of_year'] = df_features_merged.week_of_year.astype(float)

features = df_features_merged.columns.values
features = features.tolist()
features.remove('Total Load')
features.remove('time_of_day')
features.remove('weekday_name')
features.remove('month_name')

# set target, features and labels
target = 'Total Load'
X_train = df_features_merged[features]
y_train = df_features_merged[target]


# machine learning model training
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

params = {'boosting_type':'gbdt',
          'metric':'mae',
          'n_jobs':8,
          'learning_rate':0.01,
          'num_leaves': 2**8,
          'max_depth':-1,
          'tree_learner':'serial',
          'colsample_bytree': 0.9,
          'subsample_freq':1,
          'subsample':0.5,
          'max_bin':255,
          "verbosity": -1,
          "bagging_seed": 42,
          'drop_seed': 1333,
          'data_random_seed': 299,
         }

lgb_regressor = lgb.LGBMRegressor(**params, n_estimators = 700)
lgb_regressor.fit(X_train, y_train)

import joblib
# save model
joblib.dump(lgb_regressor, 'lgb.pkl')
# load model
#gbm_pickle = joblib.load('lgb.pkl')


# predict
today = datetime.date.today()

two_days_upfront = (today + datetime.timedelta(days=2)).strftime("%Y-%m-%d")
start_date = '2015-01-01'

df_wx_antwerp = get_wx_df(today.strftime("%Y-%m-%d"), two_days_upfront, freq, wx_params, request_type='forecast', city='Antwerp')
