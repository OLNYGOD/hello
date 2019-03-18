import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("2330.TW.csv")
'''
df= pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',header=None, sep='\s+')
'''
print(data.head())
type(data)
'''
data.head()
print(data.info())
len(data)
print(data.describe())
'''
print(data.columns)
data

x=data[['x1', 'x2', 'x3', 'x4', 'x5']]
y =data.estimate
print(x)

print(type(x))
len(x)

print(data.corr())
plt.matshow(data.corr())
plt.show()

x=x.values.reshape(-1,5)
y=y.values.reshape(-1,1)

print(type(x))
print(x.shape)
print(type(y))
print(y.shape)

from sklearn.linear_model import LinearRegression as LR
model = LR()
model.fit(x,y)
pre = model.predict(x)

'''
print(model)
a=model.intercept_
b=model.coef_
'''

w = data['x1'].values.reshape(-1,1)
plt.scatter(w,y)
plt.plot(w,pre, 'g-')

model.score(x,y)
