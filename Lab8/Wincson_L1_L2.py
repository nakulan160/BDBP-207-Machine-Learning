import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def main():
    data=pd.read_csv('data.csv')
    X=data.dropna(axis=1, how='all',inplace=True)
    X=data.iloc[:,2:]
    y=data.iloc[:,1].map({"B":1,"M":0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1607)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    regressor = LogisticRegression(penalty='l1',solver='liblinear')
    # regressor = LogisticRegression(penalty='l2',solver='liblinear')
    # regressor = LogisticRegression()
    regressor.fit(X_train_scaled, y_train)
    y_pred = regressor.predict(X_test_scaled)
    print(regressor.coef_)
    print(regressor.intercept_)
    print(sum(regressor.coef_[0]!=0))
    print("Accuracy: ",accuracy_score(y_test,y_pred))


if __name__=='__main__':
    main()