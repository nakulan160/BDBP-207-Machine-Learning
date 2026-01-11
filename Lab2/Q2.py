x = [[2],[1],[2]]
y = [[1],[2],[2]]
dotproduct=0
for i in range(len(x)):
    for j in range(len(x[i])):
        dotproduct += x[i][j] * y[i][j]
print(dotproduct)