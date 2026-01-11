import random
from matplotlib import pyplot as plt
xvalues=sorted([random.randint(-10,10) for _ in range(100)])
yvalues=[]
for i in xvalues:
    yvalues.append(i*i)
plt.plot(xvalues,yvalues)
ydash=[]
x1 = sorted([-5,-3,0,3,5])
for j in x1:
    ydash.append(2*j)
plt.plot(x1,ydash)
plt.show()
#the value of x at y=0 is 0