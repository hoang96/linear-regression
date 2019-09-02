import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from pandas import read_csv
#doc file
bottle = pd.read_csv (r'C:\Users\NoName\Documents\GitHub\linear-regression\bottle.csv', usecols=['Depthm','Salnty'])
operation = bottle.head(1000)

clf = linear_model.LinearRegression()
# Sử dụng "Depthm" làm biến giải thích
X = operation.loc[:, ['Depthm']].as_matrix()
 
# Sử dụng "Salnty" làm biến mục đích
y = operation['Salnty'].as_matrix()

operation.fillna(method='ffill', inplace=True)
operation.dropna(inplace=True)
 
# Tạo model suy đoán
clf.fit(X, y)
 
# Hệ số hồi quy
print(clf.coef_)
 
# Sai số
print(clf.intercept_)
 
# Score
print(clf.score(X,y))

operation = bottle.head(10000)

parameters = ['Depthm']
objective = 'Salnty'

x_real = operation[objective]
y_real = operation[parameters]

# Biểu diễn sự phân bố tập dữ liệu input
# c: color
plt.scatter(X, y, c='b')
# Đường thẳng hồi quy
plt.plot(X, clf.predict(X))
plt.show()
