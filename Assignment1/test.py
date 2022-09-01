import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 20)
y = np.sin(x)
plt.xlabel("x")
plt.ylabel("y")
xticks = np.linspace(0, 10, 11)
yticks = np.sin(xticks)
plt.xticks(xticks)
plt.yticks(yticks)
plt.plot(x, y, linestyle="-", marker="o", label="line")
plt.show()
