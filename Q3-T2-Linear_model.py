from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
coef_ = np.array([3.915504328196925,
                  17.77851215778241,
                  18.989862542516342,
                  1.7930583744217108,
                  7.192112975339136])   # weight coefficient

xlsx = pd.read_excel('Q1_attribute.xlsx')
_str = {}
_str[1] = '1 try'
_str[2] = '2 tries'
_str[3] = '3 tries'
_str[4] = '4 tries'
_str[5] = '5 tries'
_str[6] = '6 tries'
_str[7] = '7 or more tries (X)'

x_7 = [1, 2, 3, 4, 5, 6, 7]

attri1 = np.array(xlsx['vowel rate'].values)
attri2 = np.array(xlsx['rate grade'].values)
attri3 = np.array(xlsx['specific'].values)
attri4 = np.array(xlsx['similarity'].values)
attri6 = np.array(xlsx['distance'].values)
attr = np.array([attri1,
                 attri2,
                 attri3,
                 attri4,
                 attri6])
difficulty = -np.dot(coef_, attr)   # minus sign is used

accuracy = np.array([xlsx[_str[1]].values,
                     xlsx[_str[2]].values,
                     xlsx[_str[3]].values,
                     xlsx[_str[4]].values,
                     xlsx[_str[5]].values,
                     xlsx[_str[6]].values,
                     xlsx[_str[7]].values]).T
avg_accuracy = []
for i in range(len(accuracy)):
    sum = 0
    for j in range(7):
        sum += accuracy[i][j] * (j + 1)
    sum /= 100
    avg_accuracy.append(sum)
plt.scatter(difficulty,avg_accuracy)
plt.ylabel("the times of success")
plt.show()

ini_point = np.array([difficulty, attri1, attri2, attri3, attri4, attri6]).T
interval_nums = 6
length = 25 - (-5)
span = length / interval_nums

interval = []
pre = -5
cur = -5
for i in range(interval_nums):
    cur += span
    interval.append([pre,cur])
    pre = cur

colors = ['blue', 'orange', 'red', '#54B345', 'magenta', 'cyan', 'yellow', 'gray', 'green', 'purple']
categories = [[] for _ in range(6)]
for (x,att1,att3,att4,att5,att6) in ini_point:
    for (index,[i,j]) in enumerate(interval):
        if i <= x and x < j:
            categories[index].append([x,att1,att3,att4,att5,att6])
            break
avg_x = []
avg_y = []
point = []
attr_index = ['attr1','attr3','attr4','attr5','attr6']
for i in range(len(categories)):
    cnt = 0
    for tup in categories[i]:
        difficulty, att1,att3,att4,att5,att6 = tup
        att = [-att1,-att3,-att4,-att5,-att6]
        plt.plot(attr_index,att,color=colors[i])
        if cnt >= 10:
            break
        cnt += 1
    print("the color ", colors[i], ' is difficulty case ', i + 1)
plt.ylabel("the difficulty of each factor influencing",fontsize=15)
plt.xlabel("the attributes of word",fontsize=15)
plt.legend(fontsize=15)
plt.show()