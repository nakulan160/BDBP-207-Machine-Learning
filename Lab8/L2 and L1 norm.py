def hypothesis(X,theta):
    return np.dot(X,theta)

def cost_function(Xtheta,y_train,lam,theta):
     y_train=y_train.to_numpy()
     y_train=y_train.reshape(-1,1)
     diff=np.subtract(Xtheta,y_train)
     cost=np.sum(np.power(diff,2))
     cost1 = cost / 2
     l1=lam*np.sum(theta[1:])
     # l2=lam*np.sum(np.power(theta[1:],2))
     # final_cost=cost1+l2
     final_cost=cost1 + l1
     return final_cost

def find_theta(X,Xtheta,y_train,alpha,j,lam,theta):
    y_train = y_train.to_numpy()
    y_train = y_train.reshape(-1, 1)
    diff=np.subtract(Xtheta,y_train)
    s=np.sum(diff[:,0] * X[:,j])
    if j!=0:
        s+=lam*np.sign(theta[j][0])
    #     s+=2*lam*theta[j][0]
    return s*alpha

def split_data(df):
    split=int(0.7*len(df))
    df_train=df[:split]
    df_test=df[split:]
    X_train=df_train.drop(columns=['disease_score','disease_score_fluct'])
    y_train=df_train['disease_score']
    X_test=df_test.drop(columns=['disease_score','disease_score_fluct'])
    y_test=df_test['disease_score']
    return X_train, y_train, X_test, y_test

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def main():
    df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    X_train, y_train, X_test, y_test = split_data(df)
    scaler=StandardScaler()
    X=scaler.fit_transform(X_train)
    m=X.shape[0]
    X=np.c_[np.ones(m),X]
    theta=np.zeros((X.shape[1],1),dtype=float)
    lam=0.00000001
    alpha=0.001
    theta_threshold=0.001
    cost_threshold=0.001
    prev_cost=float('inf')
    a=[]
    b=[]
    theta_prev=theta.copy()
    for i in range(2000):
        Xtheta = hypothesis(X, theta)
        cost=cost_function(Xtheta,y_train,lam,theta)
        print(f"Iteration {i + 1} | Cost: {cost} | Theta: {theta.ravel()}")
        a.append(cost)
        b.append(i+1)
        for j in range(len(theta)):
            theta[j][0] -= find_theta(X, Xtheta, y_train, alpha, j,lam,theta)
        theta_change=np.max(np.abs(theta-theta_prev))
        cost_change=abs(prev_cost-cost)
        if theta_change < theta_threshold or cost_change < cost_threshold:
            print("Converged at iteration:",i+1)
            break
        theta_prev=theta.copy()
        prev_cost=cost
    X_test = scaler.transform(X_test)
    q = X_test.shape[0]
    X_test = np.c_[np.ones(q), X_test]
    y_pred=hypothesis(X_test, theta_prev)
    print("MSE", mean_squared_error(y_test,y_pred))
    print("r2", r2_score(y_test, y_pred))
    a_array = np.array(a)
    b_array = np.array(b)
    plt.plot(b_array,a_array)
    plt.xlabel('iteration')
    plt.ylabel('cost function')
    plt.xlim(0,2000)
    plt.show()

if __name__ == "__main__":
    main()

