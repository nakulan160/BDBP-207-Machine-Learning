theta=[[2],[3],[3]]
X=[[1,0,2],[0,1,1],[2,1,0],[1,1,0],[0,2,1]]
rowtheta=len(theta)
print(rowtheta)
coltheta=len(theta[0])
print(coltheta)
rowX=len(X)
print(rowX)
colX=len(X[0])
print(colX)
Xtheta=[[0 for _ in range(coltheta)] for _ in range(rowX)]
for i in range(rowX):
    for j in range(coltheta):
        for k in range(colX):
            Xtheta[i][j] += X[i][k]*theta[k][j]
for row in Xtheta:
    print(row)
