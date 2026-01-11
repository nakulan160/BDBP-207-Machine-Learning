import numpy as np
X=[[1,0,2],
   [0,1,1],
   [2,1,0],
   [1,1,1],
   [0,2,1]]
Y=X
x1bar=0
for i in range(len(X)):
    x1bar +=X[i][0]
x1bar=x1bar/len(X)
x2bar=0
for j in range(len(X)):
    x2bar+=X[j][1]
x2bar=x2bar/len(X)
x3bar=0
for k in range(len(X)):
    x3bar+=X[k][2]
x3bar=x3bar/len(X)
for i in range(len(X)):
    X[i][0]-=x1bar
for j in range(len(X)):
    X[j][1]-=x2bar
for k in range(len(X)):
    X[k][2] -=x3bar

XMINUSXBARTRANSPOSE=[[0 for _ in range(len(X))] for _ in range(len(X[0]))]
for a in range(len(X)):
   for b in range(len(X[0])):
       XMINUSXBARTRANSPOSE[b][a] = X[a][b]

XMULTIPLY=[[0 for _ in range(len(X[0]))]for _ in range(len(XMINUSXBARTRANSPOSE))]
for i in range(len(XMINUSXBARTRANSPOSE)):
    for j in range(len(X[0])):
        for k in range(len(XMINUSXBARTRANSPOSE[0])):
            XMULTIPLY[i][j] += XMINUSXBARTRANSPOSE[i][k] * X[k][j]

cov_m1=[[0 for _ in range(len(XMULTIPLY[0]))]for _ in range(len(XMULTIPLY))]
for i in range(len(XMULTIPLY)):
    for j in range(len(XMULTIPLY[0])):
        cov_m1[i][j] = XMULTIPLY[i][j]/(len(X)-1)
for row in cov_m1:
  print(row)

cov_matrix=np.cov(Y,rowvar=False)
for row in cov_matrix:
 print(row)