import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf

plt.rcParams["font.family"] = "serif"
df = pd.read_excel('Problem_C_Data_Wordle.xlsx')
data = df.copy()
data = data.set_index('index')
# data table
plt.plot(data.index,data['Number of  reported results'].values)
plt.show()
train = data.loc[:,:]
all = data.loc[:,:]
# stability test
print(sm.tsa.stattools.adfuller(train['Number of  reported results'].values))
# white noise test
print(all['Number of  reported results'])
print(acorr_ljungbox(all['Number of  reported results'], lags = [6, 9, 12, 15, 18], boxpierce=True))
# calculate ACF PACF
acf = plot_acf(all['Number of  reported results'])
plt.title('the autocorrelation plot of Number of reported')
plt.show()
pacf = plot_pacf(all['Number of  reported results'])
plt.title("the partial autocorrelation plot of Number of reported results")
plt.show()
# model training
model = sm.tsa.arima.ARIMA(train['Number of  reported results'], order=(7, 1, 7)) # order can be (1,1,0)
ar_res = model.fit()
print(ar_res.summary())
# predict the interval
forecast_result = ar_res.get_forecast(61)
forc = ar_res.forecast(61)
yhat = forecast_result.predicted_mean
yhat_conf_int = forecast_result.conf_int(alpha=0.95)
print(yhat_conf_int)
print(forc)
predict = ar_res.predict(start=train.index[0], end=train.index[-1],typ='levels')
print(predict)
rmse = ((predict - train['Number of  reported results'].values)**2).mean() ** 0.5
print("the rmse is",rmse," ")
# residual
train_array = []
predict_array = []
for i in train['Number of  reported results']:
    train_array.append(i)
for i in predict:
    predict_array.append(i)
residual = []
print(len(predict_array))
print(len(train_array))
for i in range(len(predict_array)):
    residual.append(train_array[i] - predict_array[i])
x = train.index
y = residual
mean = sum(y) / len(y)
std = (sum([(yi - mean)**2 for yi in y]) / len(y)) ** 0.5
# paint the residual plot
fig, ax = plt.subplots()
ax.bar(x, y, width=2.5, align='center', alpha=0.8)
ax.axhline(y=0, color='black', linestyle='--')
ax.axhline(y=mean, color='gray', linestyle='--', label='Mean')
ax.axhline(y=mean+2*std, color='orange', linestyle='--', label='Upper Bound')
ax.axhline(y=mean-2*std, color='orange', linestyle='--', label='Lower Bound')
ax.set_xticks(x[::30],fontsize=25)
ax.set_xlabel('date',fontsize=25)
ax.set_ylabel('Residual',fontsize=25)
ax.legend(fontsize=25)
plt.show()
# fit result
plt.plot(train.index,train['Number of  reported results'].values, label='actual')
plt.plot(train.index,predict, label='forecast')
plt.ylabel("number of reported results",fontsize=20)
plt.xlabel("date",fontsize=20)
plt.legend(fontsize=20)
plt.show()