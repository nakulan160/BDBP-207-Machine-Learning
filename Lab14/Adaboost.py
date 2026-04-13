import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RepeatedKFold

iris=load_iris()
X=pd.DataFrame(iris.data,columns=iris.feature_names)
y=pd.DataFrame(iris.target)
y=y.values.ravel()
kf=RepeatedKFold(random_state=16,n_splits=10,n_repeats=2)
a=[]
for fold,(train_idx,test_idx) in enumerate(kf.split(X)):
    X_train,X_test=X.iloc[train_idx],X.iloc[test_idx]
    y_train,y_test=y[train_idx],y[test_idx]
    model=DecisionTreeClassifier(random_state=16,max_depth=10,min_samples_split=3)
    ml=AdaBoostClassifier(estimator=model,n_estimators=20,learning_rate=0.1)
    ml.fit(X_test,y_test)
    y_pred=ml.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    a.append(acc)
print(a)
print(np.mean(a))
print(np.std(a))


