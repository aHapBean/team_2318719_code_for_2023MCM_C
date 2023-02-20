from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"

xlsx = pd.read_excel('Q1_attribute_random_trees.xlsx')
_str = {}
_str[1] = '1 try'
_str[2] = '2 tries'
_str[3] = '3 tries'
_str[4] = '4 tries'
_str[5] = '5 tries'
_str[6] = '6 tries'
_str[7] = '7 or more tries (X)'

x_7 = [1,2,3,4,5,6,7]
# independent variables
x = np.array(xlsx['index'].values)
attri1 = np.array(xlsx['vowel rate'].values)
attri2 = np.array(xlsx['is repeat'].values)
attri3 = np.array(xlsx['rate grade'].values)
attri4 = np.array(xlsx['specific'].values)
attri5 = np.array(xlsx['similarity'].values)
attri6 = np.array(xlsx['distance'].values)
w = np.array(xlsx['week'].values)
hard_rate = np.array(xlsx['rate'].values)
# dependent variables
y1 = np.array(xlsx[_str[1]].values)
y2 = np.array(xlsx[_str[2]].values)
y3 = np.array(xlsx[_str[3]].values)
y4 = np.array(xlsx[_str[4]].values)
y5 = np.array(xlsx[_str[5]].values)
y6 = np.array(xlsx[_str[6]].values)
y7 = np.array(xlsx[_str[7]].values)
# split data
data = pd.DataFrame({'attri1': attri1, 'attri2': attri2, 'attri3': attri3,'x': x,'attri6':attri6,
                     'attri4': attri4, 'attri5': attri5, 'w': w, 'hard_rate': hard_rate,
                     'y1': y1, 'y2': y2, 'y3': y3, 'y4': y4, 'y5': y5, 'y6': y6, 'y7': y7})
train_data, test_data = train_test_split(data, test_size=0.1,random_state=0)
X_train = train_data[['x','attri1', 'attri2', 'attri3', 'attri4', 'attri5','attri6', 'w', 'hard_rate']]
y_train = train_data[['y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7']]
X_test = test_data[['x','attri1', 'attri2', 'attri3', 'attri4', 'attri5', 'attri6' ,'w', 'hard_rate']]
y_test = test_data[['y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7']]
# build random forest
rf = RandomForestRegressor(n_estimators=112, random_state=42)    # n_estimators = 112
rf.fit(X_train, y_train)
# predict
y_pred = rf.predict(X_test)
for feature, importance in zip(X_train.columns, rf.feature_importances_):
    print(f"{feature}: {importance}")
# Sensitivity analysis
nums = 200
x_predict_eerie = [[418,0.6 , 1, 0.484197731,0.0908428,0,-2.23607,5,  0.09]]
cur = 3
span = x_predict_eerie[0][int(cur)] / 20 * 2
for _times in range(nums):
    x_predict_eerie[0][int(cur)] += span
    print(x_predict_eerie)
    y_predict_eerie = rf.predict(x_predict_eerie)
    _cur = y_predict_eerie.sum() / 100
    for j in range(7):
        y_predict_eerie[0][j] = y_predict_eerie[0][j] * 1 / _cur

    plt.plot(x_7,y_predict_eerie[0])
    plt.xlabel("the time needed to success",fontsize=24)
    plt.ylabel("percent / %",fontsize=25)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
plt.title("sensitivity test iterations(200 times)",fontsize=25)
plt.legend()
plt.show()
# print predicted data
for i in range(len(y_pred)):
    cur = y_pred[i].sum() / 100
    mx = 0
    for j in range(7):
        if y_test.iloc[i].values[j] >= y_test.iloc[i].values[mx]:
            mx = j
        y_pred[i][j] = y_pred[i][j] * 1 / cur
        sum += (y_pred[i][j] - y_test.iloc[i].values[j]) ** 2
    print(y_test.iloc[i].values)
    plt.plot(x_7,y_pred[i],label = "predict")
    plt.plot(x_7,y_test.iloc[i].values,label = "real")
    plt.xlabel("the time needed to success",fontsize=15)
    plt.ylabel("percent / %",fontsize=15)
    plt.legend()
    plt.show()
