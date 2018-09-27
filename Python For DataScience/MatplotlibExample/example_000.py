import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 11)
y = x ** 2

plt.plot(x, y)  # with default line color
plt.plot(x, y, 'r-')  # with RED line color
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
plt.show()  # show the plot

plt.subplot(1, 2, 1)  # row[1] column[2] item[1] plot
plt.plot(x, y, 'r')
plt.subplot(1, 2, 2)  # row[1] column[2] item[2] plot
plt.plot(y, x, 'b')
plt.show()  # show the plot

# OO
fig = plt.figure()  # Create empty canvas
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(x, y)
axes.set_xlabel('X Label')
axes.set_ylabel('Y Label')
axes.set_title('Title')
plt.show()  # show the plot
