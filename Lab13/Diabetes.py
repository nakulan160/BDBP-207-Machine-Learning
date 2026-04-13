import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

data = load_diabetes()
X=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target)
y=y.fillna(y.mean())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)

scalar=StandardScaler()
X_train_scaled=scalar.fit_transform(X_train)
X_test_scaled=scalar.transform(X_test)

model_lasso=ElasticNet(alpha=0.1,max_iter=10000,l1_ratio=0.5,random_state=123)
cv_lasso=cross_val_score(model_lasso,X_train_scaled,y_train,cv=10,scoring='neg_mean_squared_error')
print("Elastinet_mean",cv_lasso.mean())
print("Elastinet_std",cv_lasso.std())
model_lasso.fit(X_train_scaled,y_train)
y_pred_lasso=model_lasso.predict(X_test_scaled)
print("r2_lasso",r2_score(y_test,y_pred_lasso))
print("mse_lasso",mean_squared_error(y_test,y_pred_lasso))


rf=RandomForestRegressor(n_estimators=300,random_state=123,n_jobs=-1,max_depth=10,min_samples_split=10,min_samples_leaf=2)
rf.fit(X_train_scaled,y_train)
score=cross_val_score(rf,X_train,y_train,cv=10,scoring='neg_mean_squared_error')
print("mean_rf:",score.mean())
print("std_rf",score.std())
rf.fit(X_train,y_train)
tree=rf.estimators_[0]
plt.figure(figsize=(15,8))
plot_tree(tree,feature_names=X.columns,filled=True,rounded=True,fontsize=18)
plt.show()
y_pred=rf.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mse_rf",mse)
print("r2_rf",r2)

bagging=DecisionTreeRegressor(max_depth=80,random_state=123)
bag_cv=cross_val_score(bagging,X_train,y_train,cv=10,scoring='neg_mean_squared_error')
print("bag_cv_mean",bag_cv.mean())
print("bag_cv_std",bag_cv.std())
bagging.fit(X_train,y_train)
y_pred=bagging.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mse_bagging",mse)
print("r2_bagging",r2)
plot=plot_tree(bagging,feature_names=X.columns,filled=True,rounded=True,fontsize=18)
plt.show()
