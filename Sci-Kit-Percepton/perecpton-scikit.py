from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score



iris = datasets.load_iris()

X = iris.data[:, [2,3]]
y = iris.target

print("***")
print(X.shape)
print(y.shape)
print("***")


print(np.unique(y))


X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)

sc = StandardScaler()

sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)



print(X_test_std.shape)
print(y.shape)

ppn = Perceptron(max_iter=40, eta0=0.01, random_state=0, tol=None)

ppn.fit(X_train_std, y_train)


y_pred = ppn.predict(X_test_std)
print("Missclasified samples: ", (y_test != y_pred).sum())

print("Accuracy: ", accuracy_score(y_test, y_pred))