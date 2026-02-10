import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1607)
    return X_train, X_test, y_train, y_test

def scale_data(X_train,X_test):
    scaler = StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    return X_train_scaled,X_test_scaled

def test_model(model,X_test_scaled,y_test):
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse,r2

def main():
    df = pd.read_csv('data.csv')
    df=df.dropna(axis=1, how='all', inplace=False)
    X=df.drop(columns=["id","diagnosis"])
    y=df['diagnosis'].map({'M':1,'B':0})
    X_train, X_test, y_train, y_test = split_data(X,y)
    X_train_scaled, X_test_scaled = scale_data(X_train,X_test)
    model = LogisticRegression()
    model.fit(X_train_scaled,y_train)
    mse,r2 = test_model(model,X_test_scaled,y_test)
    print("MSE",mse)
    print("r2",r2)
    print("Theta values",model.coef_)


if __name__ == '__main__':
    main()