import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 11)
y = x ** 2

fig = plt.figure()  # Create empty canvas
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes1.set_title('Larger plot')

axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])
axes2.set_title('Smaller plot')

axes1.plot(x, y)
axes2.plot(y, x)
# plt.show()

# fig, axes = plt.subplot(nrows=1, ncols=2)
# # for current_ax in axes:
# #     current_ax.plot(x, y)
# axes[0].plot(x, y)
# axes[0].set_title('First Plot')
# axes[1].plot(y, x)
# axes[1].set_title('Second Plot')
# plt.show()

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.plot(x, x ** 2, label='X Squared')
ax.plot(x, x ** 3, label='X Cubed')

ax.legend(loc=0)
plt.show()
