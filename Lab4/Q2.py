import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler,MinMaxScaler



def load_data():
    california_housing = fetch_california_housing(as_frame=True)
    X = california_housing.data
    y = california_housing.target
    return X,y

def main():
    X,y = load_data()
    print(X.columns)
    print(y.columns)


if __name__ == "__main__":
    main()