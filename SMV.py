import pandas as pd

from sklearn import svm
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

################################## Prepare data
path="C:/Users/vanes/OneDrive - Universitat de Barcelona/8e semestre/ML-EELS/"

ds = path + '/Mn_Fe_dataset.pkl'
lb = path + '/Mn_Fe_labels.pkl'

X = pd.read_pickle(ds)
y = pd.read_pickle(lb)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 0)

################################# Model
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy_score(y_test, y_pred) # accuracy
