# K-fold cross validation. Implement for K = 10.
# Implement from scratch, then, use scikit-learn methods.
import statistics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# def split_K_Fold(X, y, K):
#     X = X.reset_index(drop=True)
#     y = y.reset_index(drop=True)
#     fold_size = len(X) // K
#     folds = []
#     for i in range(K):
#         start = i * fold_size
#         end = (i + 1) * fold_size
#         X_test = X.iloc[start:end]
#         y_test = y.iloc[start:end]
#         X_train = pd.concat([X.iloc[:start], X.iloc[end:]])
#         y_train = pd.concat([y.iloc[:start], y.iloc[end:]])
#         folds.append((X_train, X_test, y_train, y_test))
#     return folds

def scale_data(X_train,X_test):
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    return X_train_scaled,X_test_scaled

def main():
    # data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    cf=fetch_california_housing()
    datax=cf.data
    X=pd.DataFrame(datax,columns=cf.feature_names)
    y=cf.target
    y=pd.Series(y)
    # split_data=split_K_Fold(X,y,10)
    kf=KFold(n_splits=10,shuffle=True,random_state=42)
    iter = []
    rsquare = []
    meansquare = []
    for fold, (train_idx,test_idx) in enumerate(kf.split(X),start=1):
                       X_train=X.iloc[train_idx]
                       y_train=y.iloc[train_idx]
                       X_test=X.iloc[test_idx]
                       y_test=y.iloc[test_idx]
        # X_train, X_test, y_train, y_test = split_data[i]
                       X_train_scaled, X_test_scaled=scale_data(X_train,X_test)
                       model = LinearRegression()
                       model.fit(X_train_scaled, y_train)
                       y_pred = model.predict(X_test_scaled)
                       r2=r2_score(y_test, y_pred)
                       mse=mean_squared_error(y_test, y_pred)
                       iter.append(fold)
                       rsquare.append(r2)
                       meansquare.append(mse)
                       print(f"fold {fold}: r2 {r2}: mse {mse}")
        # print(f'Fold{i+1} r2:{r2} mean_squared_error:{mse} ')
    # print("rsquare mean:", statistics.mean(rsquare))
    # print("rsquare sd:", statistics.stdev(rsquare))
    # print("mean squared error mean:", statistics.mean(meansquare))
    # print("mse stdev:", statistics.stdev(meansquare))
    plt.plot(iter,rsquare,label='r2')
    plt.xlabel("itertion")
    plt.ylabel("r2")
    plt.legend()
    plt.show()
    plt.plot(iter,meansquare,label='mse')
    plt.xlabel("iteration")
    plt.ylabel("mse")
    plt.legend()
    plt.show()


if __name__=='__main__':
    main()