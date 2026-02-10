import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def hypothesis(X_train,theta):
    return np.dot(X_train,theta)

def cost_function(Xtheta,y_train):
     y_train=y_train.to_numpy()
     y_train=y_train.reshape(-1,1)
     diff=np.subtract(Xtheta,y_train)
     cost=np.sum(np.power(diff,2))
     cost_function=cost/2
     return cost_function

def find_theta(X_train,Xtheta,y_train,alpha,j,random_value):
    y_train = y_train.to_numpy()
    y_train = y_train.reshape(-1, 1)
    Xtheta=Xtheta[random_value,0]
    y_train=y_train[random_value,0]
    diff=np.subtract(Xtheta,y_train)
    s=diff* X_train[random_value,j]
    salpha = s * alpha
    return salpha

def split_data(df):
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
    y_test=df_test['disease_score'].values.reshape(-1,1)
    return X_train, y_train, X_test, y_test

def main():
    df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    X_train, y_train, X_test, y_test = split_data(df)
    theta=np.zeros((len(X_train[0]),1),dtype=float)
    alpha=0.01
    theta_threshold=0.001
    cost_threshold=0.001
    prev_cost=float('inf')
    a=[]
    b=[]
    theta_prev=theta.copy()
    for i in range(20000):
        Xtheta = hypothesis(X_train, theta)
        cost=cost_function(Xtheta,y_train)
        print(f"Iteration {i + 1} | Cost: {cost} | Theta: {theta.ravel()}")
        a.append(cost)
        b.append(i+1)
        random_value = np.random.randint(0, len(X_train))
        for j in range(len(theta)):
            # random_value = np.random.randint(0, len(X_train))
            print("Random value is: ", random_value)
            theta[j][0] -= find_theta(X_train, Xtheta, y_train, alpha, j,random_value)
        theta_change=np.max(np.abs(theta-theta_prev))
        cost_change=abs(prev_cost-cost)
        if theta_change < theta_threshold or cost_change < cost_threshold:
            print("Converged at iteration:",i+1)
            break
        theta_prev=theta.copy()
        prev_cost=cost
    y_pred=hypothesis(X_test, theta_prev)
    # print("MSE",mean_squared_error(y,Xtheta))
    print("MSE", mean_squared_error(y_test,y_pred))
    # print("r2",r2_score(y,Xtheta))
    print("r2", r2_score(y_test, y_pred))
    a_array = np.array(a)
    b_array = np.array(b)
    plt.plot(b_array,a_array)
    plt.xlabel('iteration')
    plt.ylabel('cost function')
    plt.xlim(0,20000)
    plt.show()

if __name__ == "__main__":
    main()

