def hypothesis(X,theta):
    rowx=len(X)
    colx=len(X[0])
    rowtheta=len(theta)
    coltheta=len(theta[0])
    Xtheta=[[0 for _ in range(coltheta)]for _ in range(rowx)]
    for i in range(rowx):
        for j in range(coltheta):
            for k in range(colx):
                Xtheta[i][j]+=X[i][k]*theta[k][j]
    return Xtheta

def cost_function(y,Xtheta):
    Xthetaminusy = []
    for i in range(len(Xtheta)):
        Xthetaminusy.append(Xtheta[i][0]-y[i][0])
    sum=0
    for j in range(len(Xthetaminusy)):
        sum+=Xthetaminusy[j]*Xthetaminusy[j]
    cost_function=sum/2
    return cost_function

def find_theta(X,Xtheta,y,alpha,j):
   s=0
   for i in range(len(X)):
       error=(Xtheta[i][0]-y[i][0])
       s=s+error*X[i][j]
   return s*alpha


# input
theta = [[0], [0]]
X = [[1, 2], [1, 3]]
y = [[3], [5]]
alpha = 0.001
def main():
   for i in range(200):
        Xtheta = (hypothesis(X, theta))
        print(i+1)
        print("Cost function:", cost_function(y, Xtheta))
        for j in range(len(theta)):
            theta[j][0] -= find_theta(X,Xtheta, y, alpha,j)
        print(theta)
   X1 = [[1, 2.5]]
   print(hypothesis(X1, theta))



if __name__ == "__main__":
    main()


    # Xthetaminussy=[[0 for _ in range(1)] for _ in range(len(Xtheta))]
    # for i in range(len(Xtheta)):
    #     Xthetaminussy[i][0]=(Xtheta[i][0] - y[i][0])
    # xtmt=[[0 for _ in range(len(Xthetaminussy))]for _ in range(len(Xthetaminussy[0]))]
    # for a in range(len(xtmt[0])):
    #     for b in range(len(xtmt)):
    #         xtmt[b][a]=Xthetaminussy[a][b]
    # htx=[]
    # for c in range(len(xtmt[0])):
    #     htx.append(xtmt[0][c]*X[i][0])
    # s=0
    # for d in range(len(htx)):
    #     s+=htx[d]
    # salpha=s*alpha
    # new_theta0=0-salpha
    # return new_theta0

# def find_theta1(Xtheta,y,alpha):
#     Xthetaminussy=[[0 for _ in range(1)] for _ in range(len(Xtheta))]
#     for i in range(len(Xtheta)):
#         Xthetaminussy[i][0]=(Xtheta[i][0] - y[i][0])
#     xtmt=[[0 for _ in range(len(Xthetaminussy))]for _ in range(len(Xthetaminussy[0]))]
#     for a in range(len(xtmt[0])):
#         for b in range(len(xtmt)):
#             xtmt[b][a]=Xthetaminussy[a][b]
#     htx=[]
#     for c in range(len(xtmt[0])):
#         htx.append(xtmt[0][c]*X[c][1])
#     s=0
#     for d in range(len(htx)):
#         s+=htx[d]
#     salpha=s*alpha
#     new_theta1=0-salpha
#     return new_theta1

# def update_theta():
#     update_theta0=



