import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data(filename):
    df = pd.read_csv(filename)
    #X=df.drop("disease_score","disease_fluct",axis=1)
    #y=df["disease_score"]
    X=df.iloc[:,:5]
    y=df.iloc[:,6]
    return X,y
def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
def scale_data(X_train,X_test):
    scaler = StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    return X_train_scaled,X_test_scaled
def test_data(model,X_test_scaled,y_test):
    y_pred=model.predict(X_test_scaled)
    mse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    return mse,r2

def main():
    X,y=load_data('simulated_data_multiple_linear_regression_for_ML.csv')
    X_train, X_test, y_train, y_test = split_data(X,y)
    X_train_scaled, X_test_scaled = scale_data(X_train,X_test)
    model=LinearRegression()
    model.fit(X_train_scaled, y_train)
    mse,r2=test_data(model,X_test_scaled,y_test)
    print("MSE",mse)
    print("r2",r2)

if __name__=="__main__":
    main()
