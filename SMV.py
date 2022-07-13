from sklearn import svm

X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
regr = svm.SVR()
regr.fit(X, y)
SVR()
regr.predict([[1, 1]])
array([1.5])
