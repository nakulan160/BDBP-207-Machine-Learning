import random
import matplotlib.pyplot as plt
xvalues=[random.randint(-100,100) for _ in range(100)]
print(xvalues)
yvalues=[]
for i in xvalues:
    yvalues.append((2*(i)+3))
print(yvalues)
plt.plot(xvalues, yvalues)
plt.show()
