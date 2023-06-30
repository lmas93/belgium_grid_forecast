from elia_data import get_elia_data_ods001
import pandas as pd
import datetime
import matplotlib.pyplot as plt

data = get_elia_data_ods001()
data.drop('Resolution code', axis=1, inplace=True)

# standardise to utc due to daylight saving & set date as index
data['DateTime'] = pd.to_datetime(data['DateTime'], utc=True)
data.set_index('DateTime', inplace=True)
data = data.sort_index()

today = datetime.date.today()
three_days_back = (today - datetime.timedelta(days=3)).strftime("%Y-%m-%d")


prediction = pd.read_csv('predictions_hist.csv')
prediction['DateTimeUTC'] = pd.to_datetime(prediction['DateTimeUTC'], utc=True)
prediction.set_index('DateTimeUTC', inplace=True)

plt.figure()
plt.plot(data[three_days_back:].index, data[three_days_back:]['Total Load'], label = 'Actual')
for i, date in enumerate(prediction['model_run_date'].unique()):
    plt.plot(prediction.query('model_run_date == @date').index, prediction.query('model_run_date == @date')['prediction'], label = 'pred_'+str(i))
plt.legend()
plt.ylabel('Total Load (MWh)')
plt.xlabel('Date Time UTC')
plt.show()