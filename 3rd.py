import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

bitcoin = pd.read_csv('data/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')
bitcoin['Dates'] = pd.to_datetime(bitcoin['Timestamp'], unit='s')

plt.figure(figsize=(12, 7))
plt.plot(bitcoin["Dates"], bitcoin["Weighted_Price"], color='red', lw=2)
plt.title("Bitcoin Price over time", size=25)
plt.xlabel("Time", size=20)
plt.ylabel("$ Price", size=20)
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

plt.figure(figsize=(12, 7))
plt.plot(bitcoin["Dates"], bitcoin["Volume_(Currency)"], color='blue', lw=2)
plt.title("Bitcoin Volume over time", size=25)
plt.xlabel("Time", size=20)
plt.ylabel("Volume", size=20)
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

bitcoin.isnull().sum()
# bitcoin['Dates'] = pd.to_datetime(bitcoin['Timestamp'], unit='s')
bitcoin.head()
bitcoin.dropna(inplace=True)
required_features = ['Open', 'High', 'Low', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']
output_label = 'Close'

x_train, x_test, y_train, y_test = train_test_split(bitcoin[required_features], bitcoin[output_label], test_size=0.3)

model = LinearRegression()
model.fit(x_train, y_train)
model.score(x_test, y_test)

future_set = bitcoin.shift(periods=30).tail(30)
prediction = model.predict(future_set[required_features])

plt.figure(figsize=(12, 7))
plt.plot(bitcoin["Dates"][-400:-60], bitcoin["Weighted_Price"][-400:-60], color='goldenrod', lw=2)
plt.plot(future_set["Dates"], prediction, color='black', lw=2)
myFmt = mdates.DateFormatter('%m-%d %H:%M:%S')

ax = plt.gca()
ax.tick_params(axis='x', labelrotation=45)
ax.xaxis.set_major_formatter(myFmt)

plt.title("Bitcoin Price over time", size=25)
plt.xlabel("Time (Year 2021)", size=20)
plt.subplots_adjust(bottom=0.138)
plt.ylabel("$ Price", size=20)
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.show()

