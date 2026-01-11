import random
import matplotlib.pyplot as plt
xvalues=sorted([random.randint(-10,10) for _ in range(100)])
yvalues=[]
print(xvalues)
for i in xvalues:
    yvalues.append((2*(i*i))+(3*i)+4)
print(yvalues)
plt.plot(xvalues,yvalues)
plt.show()