import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def hypothesis(X,theta):
    return np.dot(X,theta)

def split_data_and_scale(df):
    split=int(0.7*len(df))
    df_train=df[:split]
    df_test=df[split:]
    X_train=df_train.drop(columns=['disease_score','disease_score_fluct'])
    m=X_train.shape[0]
    X_train=np.c_[np.ones(m), X_train]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    y_train=df_train['disease_score']
    #y=df['disease_score_fluct'].values.reshape(-1,1)
    X_test=df_test.drop(columns=['disease_score','disease_score_fluct'])
    n= X_test.shape[0]
    X_test = np.c_[np.ones(n), X_test]
    X_test=scaler.transform(X_test)
    y_test=df_test['disease_score']
    return X_train, y_train, X_test, y_test


def normal_equations(X,Y):
    XT=X.transpose()
    XTX=np.dot(XT,X)
    XTXI=np.linalg.inv(XTX)
    XTXIXT=np.dot(XTXI,XT)
    XTXIXTY=np.dot(XTXIXT,Y)
    return XTXIXTY

def main():
    df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    X_train, y_train, X_test, y_test=split_data_and_scale(df)
    theta_value=normal_equations(X_train,y_train)
    print("The value of theta is:",theta_value)
    y_pred=hypothesis(X_test,theta_value)
    print("The rsquare value is ",r2_score(y_test,y_pred))
    print("The mean squared error is ",mean_squared_error(y_test,y_pred))

if __name__=='__main__':
    main()
