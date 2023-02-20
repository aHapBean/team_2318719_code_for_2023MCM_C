import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
xlsx = pd.read_excel('Problem_C_Data_Wordle.xlsx')
# paint plot
for i in range(4, 7):
    _str = ''
    if i == 1:
        _str = str(i) + ' try'
    else:
        _str = str(i) + ' tries'
    colo = ''
    if i % 3 == 1:
        color = '#1f77b4' # blue
    elif i % 3 == 2:
        color = '#2ca02c' # green
    else:
        color = '#8c564b' # brown
    plt.plot(xlsx['Date'].values, xlsx[_str].values, label=_str,color=color)
    df_3tries = xlsx[['Date', _str]]
    mean_3tries = df_3tries[_str].mean()
    std_3tries = df_3tries[_str].std()

    upper_3tries = mean_3tries + 1.96 * std_3tries
    lower_3tries = mean_3tries - 1.96 * std_3tries
    plt.fill_between(df_3tries['Date'], upper_3tries, lower_3tries, alpha=0.3, color=color)
plt.plot(xlsx['Date'].values, xlsx['7 or more tries (X)'].values,label="7 or more tries (X)",color='blue')
df_3tries = xlsx[['Date','7 or more tries (X)']]
mean_3tries = df_3tries['7 or more tries (X)'].mean()
std_3tries = df_3tries['7 or more tries (X)'].std()

upper_3tries = mean_3tries + 1.96 * std_3tries
lower_3tries = mean_3tries - 1.96 * std_3tries
plt.fill_between(df_3tries['Date'], upper_3tries, lower_3tries, alpha=0.3, color="blue")

x_ticks = xlsx['Date'][::30]
plt.xticks(x_ticks)
plt.legend(fontsize=20)
plt.show()


