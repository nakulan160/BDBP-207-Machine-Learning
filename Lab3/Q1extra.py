def hypothesis(X,theta):
    return np.dot(X,theta)

def cost_function(Xtheta,y):
    diff=np.subtract(Xtheta,y)
    cost=np.sum(np.power(diff,2))
    cost_function=cost/2
    return cost_function

def find_theta(X,Xtheta,y,alpha,j):
    diff=np.subtract(Xtheta,y)
    s=np.sum(diff[:,0] * X[:,j])
    salpha = s * alpha
    return salpha

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def main():
    df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    X=df.drop(columns=['disease_score','disease_score_fluct'])
    scaler=StandardScaler()
    X=scaler.fit_transform(X)
    #y=df['disease_score_fluct'].values.reshape(-1,1)
    y=df['disease_score'].values.reshape(-1,1)
    m=X.shape[0]
    X=np.c_[np.ones(m),X]
    theta=np.zeros((X.shape[1],1),dtype=float)
    alpha=0.001
    theta_threshold=0.001
    cost_threshold=0.001
    prev_cost=float('inf')
    a=[]
    b=[]
    theta_prev=theta.copy()
    for i in range(2000):
        Xtheta = hypothesis(X, theta)
        cost=cost_function(Xtheta,y)
        print(f"Iteration {i + 1} | Cost: {cost} | Theta: {theta.ravel()}")
        a.append(cost)
        b.append(i+1)
        for j in range(len(theta)):
            theta[j][0] -= find_theta(X, Xtheta, y, alpha, j)
        theta_change=np.max(np.abs(theta-theta_prev))
        cost_change=abs(prev_cost-cost)
        if theta_change < theta_threshold or cost_change < cost_threshold:
            print("Converged at iteration:",i+1)
            break
        theta_prev=theta.copy()
        prev_cost=cost
    print("MSE",mean_squared_error(y,Xtheta))
    print("r2",r2_score(y,Xtheta))
    a_array = np.array(a)
    b_array = np.array(b)
    plt.plot(b_array,a_array)
    plt.xlabel('iteration')
    plt.ylabel('cost function')
    plt.xlim(0,2000)
    plt.show()

if __name__ == "__main__":
    main()

