import numpy as np
import pandas as pd
df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
# print(df.columns)
X=df.drop(columns=['disease_score','disease_score_fluct'])
y=df['disease_score_fluct']
# X=np.array([1,1,2])
m=X.shape[0]
X=np.c_[np.ones(m),X]
theta=np.zeros(X.shape[1])
alpha=0.1
# y=3

def hypothesis(X,theta):
    return np.dot(X,theta)

def find_theta(theta,X,y,alpha):
     h=hypothesis(X,theta)
     error=h-y
     theta=theta - (alpha*error*X)
     return theta
a=find_theta(theta,X,y,alpha)
print(a)

    # Xtheta= 0
    # for i in range(len(X)):
    #     for j in range(len(X[i])):
    #         Xtheta= X[i][j] * theta[i][j]
    # return Xtheta


#
# theta=find_theta(theta,X,y,alpha)
# print(theta)


#
# def cost_functiom(y,Xtheta):
#     cost_function=((Xtheta-y)**2)/2
#     return cost_function
#
# def derivative():
#     ptheta0=hypothesis()*x0
#     ptheta1=hypothesis()*x1
#     return ptheta0,ptheta1




