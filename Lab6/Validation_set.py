import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1607)
    return X_train, X_test, y_train, y_test

def main():
    data=pd.read_csv("Fishers maket.csv")
    X=data.iloc[:,2:]
    y=data.iloc[:,1]
    X_train, X_test, y_train, y_test = split_data(X,y)
    val_X_train, val_X_test, val_y_train, val_y_test = split_data(X_train, y_train)
    scaler = StandardScaler()
    val_X_train_scaled= scaler.fit_transform(val_X_train)
    val_X_test_scaled = scaler.transform(val_X_test)
    regressor = LinearRegression()
    regressor.fit(val_X_train_scaled, val_y_train)
    y_Val_pred = regressor.predict(val_X_test_scaled)
    print("Mean squared error for val set: ",mean_squared_error(val_y_test, y_Val_pred))
    print("R2 score for val set: ",r2_score(val_y_test, y_Val_pred))
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    regressor = LinearRegression()
    regressor.fit(X_train_scaled, y_train)
    y_pred = regressor.predict(X_test_scaled)
    print("Mean squared error: ",mean_squared_error(y_test, y_pred))
    print("R2 score: ",r2_score(y_test, y_pred))


if __name__ == "__main__":
    main()