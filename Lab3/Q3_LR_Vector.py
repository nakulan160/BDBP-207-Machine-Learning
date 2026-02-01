import numpy as np

def hypothesis(X,theta):
    return np.dot(X,theta)

def cost_function(y,Xtheta):
    diff=np.subtract(Xtheta,y)
    cost=np.sum(np.power(diff,2))
    cost_function=cost/2
    return cost_function

def find_theta(X,Xtheta,y,alpha,j):
    diff=np.subtract(Xtheta,y)
    s=np.sum(diff[:,0] * X[:,j])
    salpha = s * alpha
    return salpha

theta=np.array([[0.0],[0.0]]) #or np.zeroes((2,1),dtype=float)
y=np.array([[3],[5]])
X=np.array([[1,2],[1,3]])
alpha=0.001

def main():
  for i in range(400):
    Xtheta = hypothesis(X, theta)
    print(i+1, "Cost:", cost_function(y, Xtheta))
    for j in range(len(theta)):
        theta[j][0] -= find_theta(X, Xtheta, y, alpha, j)
    print("Final theta:", theta)
  X1 = np.array([[1.0, 2.5]])
  print("Prediction:", hypothesis(X1, theta))



if __name__ == "__main__":
    main()






# def find_theta1(Xtheta,y,alpha):
#     diff=np.subtract(Xtheta,y)
#     X2=X[:,1].reshape(1,-1)
#     s=np.sum(np.dot(X2,diff))
#     salpha=s*alpha
#     new_theta1 = 0-salpha
#     return new_theta1


