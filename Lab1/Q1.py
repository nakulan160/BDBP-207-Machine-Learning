A=[[1,2,3],[4,5,6]]
nrowA=len(A)
ncolA=len(A[0])
ATranspose=[[0 for _ in range(nrowA)] for _ in range(ncolA)]
for a in range(nrowA):
   for b in range(ncolA):
       ATranspose[b][a] = A[a][b]
ncolB=len(ATranspose[0])
C=[[0 for _ in range(ncolB)]for _ in range(nrowA)]

for i in range(nrowA):
    for j in range(ncolB):
        for k in range(ncolA):
            C[i][j] += A[i][k] * ATranspose[k][j]
for row in C:
  print(row)