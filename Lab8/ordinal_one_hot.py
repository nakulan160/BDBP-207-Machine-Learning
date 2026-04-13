import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,Lasso,Ridge
from sklearn.metrics import accuracy_score


def main():
    data1=pd.read_csv('Titanic_train.csv')
    data2=pd.read_csv('Titanic_test.csv')
    data=pd.concat([data1,data2])
    data=data.drop(columns=['PassengerId',"Name","Ticket","Cabin"],inplace=False,axis=1)
    data["Sex"]=data['Sex'].map({'male':0, 'female':1})
    data["Age"]=data['Age'].fillna(data['Age'].median())
    data["Pclass"]=data['Pclass'].map({3:1, 2:2, 1:3})
    data["Fare"]=data["Fare"].fillna(data['Fare'].median())
    data["Embarked"]=data['Embarked'].fillna(data["Embarked"].mode()[0])
    for val in data["Embarked"].unique():
        data[f'Embarked{val}']=(data["Embarked"]==val).astype(int)
    data=data.drop(columns=["Embarked"],inplace=False,axis=1)
    data.loc[data["Survived"].isna(),"Survived"]=np.random.randint(0,2,size=data["Survived"].isna().sum())
    X=data.drop(columns=["Survived"])
    y=data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    regressor = LogisticRegression(penalty='l1',solver='liblinear')
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print("Accuracy: ",accuracy_score(y_test, y_pred))
    print(regressor.coef_)
    print(regressor.intercept_)
    print(sum(regressor.coef_[0]!=0))





if __name__ == '__main__':
    main()