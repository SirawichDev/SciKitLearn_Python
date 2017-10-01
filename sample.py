import sklearn.datasets as ds
import numpy as np

iris = ds.load_iris()
X = iris.data[:,[2,3]]

y = iris.target
print(np.unique(y))
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)