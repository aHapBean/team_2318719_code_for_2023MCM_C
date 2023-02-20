import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
plt.rcParams["font.family"] = "serif"
coef_ = np.array([3.915504328196925,
                  17.77851215778241,
                  18.989862542516342,
                  1.7930583744217108,
                  7.192112975339136])

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
attr = np.array([attri1,attri2,attri3,attri4,attri6])

difficulty = -np.dot(coef_, attr)

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

ini_point = np.array([difficulty, xlsx['index'].values]).T
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

colors = ['blue', 'orange', 'red', 'gray', 'magenta', 'cyan', 'yellow', 'gray', 'green', 'purple']

categories = [[] for _ in range(6)]
for (x,y) in ini_point:
    for (index,[i,j]) in enumerate(interval):
        if i <= x and x < j:
            categories[index].append([x,y])
            break
avg_x = []
avg_y = []
gain_factor = [1.2,1.1,1.0,0.9,0.8,0.7]
gain_factor.reverse()

point = []
for i in range(len(categories)):
    x = [coord[0] for coord in categories[i]]
    y = [coord[1] for coord in categories[i]]
    avg_x.append(np.array(x).mean())
    avg_y.append(np.array(y).mean())
    for j in range(len(x)):
        point.append(np.array([x[j], y[j]]))
    plt.scatter(x,y,color=colors[i])
# visualize
point = np.array(point)
plt.ylim([0, 700])
corr_matrix = np.corrcoef(point[:,0], point[:,1])
plt.ylabel("the index of time",fontsize=15)
plt.xlabel("the words' difficulty",fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
print(corr_matrix)
plt.show()
# normalize
def nnormalize(tmp,Flag:bool=False):
    mean_y = np.mean(tmp[:, 1])
    std_y = np.std(tmp[:, 1])
    if Flag:
        std_y += 0.1
    normalized_y = (tmp[:, 1] - mean_y) / std_y
    return np.column_stack((tmp[:, 0], normalized_y))
tmp1 = np.column_stack((xlsx['index'].values, ini_point[:, 0]))
tmp2 = np.column_stack((xlsx['index'].values, ini_point[:, 1]))

tmp1 = nnormalize(tmp1)
tmp2 = nnormalize(tmp2,True)

# paint the relationship
plt.plot(xlsx['index'].values,tmp1[:,1],label="the words' difficulty")
plt.plot(xlsx['index'].values,tmp2[:,1],label="the average time needed to success")
plt.xlabel("the index of time",fontsize=21)
plt.ylabel("the value after normalizing",fontsize=21)
plt.legend(fontsize=21)
plt.show()

xx = xlsx['index'].values
plt.scatter(xx,tmp1[:,1],label="the words' difficulty")
plt.ylabel("the index of time",fontsize=15)
plt.xlabel("the words' difficulty",fontsize=15)
plt.legend(fontsize=15)
plt.show()

np.random.seed(0)
x = point[:,0]
y = point[:,1]
# linear fit
reg = LinearRegression()
reg.fit(x.reshape(-1, 1), y)
# print
print("Slope:", reg.coef_[0])
print("Intercept:", reg.intercept_)
y_pred = reg.predict(x.reshape(-1, 1))
# paint
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()
from sklearn.metrics import r2_score
r2 = r2_score(y, y_pred)
print('R-squared:', r2)