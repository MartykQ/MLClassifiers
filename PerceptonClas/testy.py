import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Perce import Perceptron

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/iris/iris.data')

print(df)

y = df.iloc[0:99, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:99, [0,1, 2]].values

print(y)
print(X)
"""
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')

plt.scatter(X[50:99, 0], X[50:99, 1],
            color='blue', marker='x', label='versicolor')

"""

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

print(ppn.predict([1, 99, 3]))
