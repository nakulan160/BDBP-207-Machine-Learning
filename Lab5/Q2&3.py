# #z=theta transpose x
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# def sigmoid(z):
#     return 1/(1+np.exp(-z))
#
# z=np.linspace(-3,3,100)
# y=sigmoid(z)
# plt.xlabel('z')
# plt.ylabel('y')
# plt.plot(z,y)
# plt.show()

import math


n = 10
p = 0.5

for x in range(n + 1):
    nCx = math.comb(n, x)
    prob = nCx * (p ** x) * ((1 - p) ** (n - x))
    print(f"x = {x}, P(X=x) = {prob}")


