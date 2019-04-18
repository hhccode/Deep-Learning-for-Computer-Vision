import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.hlines(y=2/45, xmin=0, xmax=5, color='green', alpha=0.5, label='P(x|w1)P(w1)')
ax.vlines(x=0, ymin=0, ymax=2/45, color='green', alpha=0.5)
ax.vlines(x=5, ymin=0, ymax=2/45, color='green', alpha=0.5)

ax.hlines(y=1/9, xmin=2, xmax=9, color='red', alpha=0.5, label='P(x|w2)P(w2)')
ax.vlines(x=2, ymin=0, ymax=1/9, color='red', alpha=0.5)
ax.vlines(x=9, ymin=0, ymax=1/9, color='red', alpha=0.5)

ax.set_ylim([0, 0.15])
plt.xticks([i for i in range(10)])
ax.legend()

plt.show()