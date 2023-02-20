import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
xlsx = pd.read_excel('Q1_attribute_pre_move.xlsx')
numerator = xlsx['Number in hard mode'].values.copy()
denominator = xlsx['Number of reported results'].values.copy()
words = xlsx['Word'].values.copy()

rate = []
for i in range(len(numerator)):
    rt = numerator[i] / denominator[i]
    rate.append(rt)
X = xlsx[['vowel rate','is repeat','rate grade','specific','similarity','index','index_sq']]
y = rate.copy()
X = np.array(X)
X = np.hstack([np.ones((len(X), 1)), X])
# improved Ridge Regression
alpha = 0.4
XTX = np.dot(X.T,X)
XTy = np.dot(X.T,y)
w = np.dot(np.linalg.inv(XTX + alpha * np.eye(X.shape[1])), XTy)
y_pred = np.dot(X,w)
residuals = y - y_pred

plt.plot(xlsx['index'].values,y_pred,label="predict")
plt.plot(xlsx['index'].values,rate,label="real")

x_ticks = xlsx['index'][::20]
plt.xticks(x_ticks,fontsize=16)
plt.ylabel("the percent of participants in hard mode",fontsize=20)
plt.xlabel("index",fontsize=20)
plt.yticks(fontsize=16)
plt.legend(fontsize=20)
plt.show()
# print regression parameters
print(w)


