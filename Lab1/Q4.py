import math
import matplotlib.pyplot as plt
import random
xvalues=sorted([random.randint(-100,100) for _ in range(100)])
yvalues=[]
for i in xvalues:
    G=(math.e**(-(i*i)/(2*225)))/(15*(math.sqrt(2*math.pi)))
    yvalues.append(G)
plt.plot(xvalues,yvalues)
plt.show()