from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

npoints = 8
n_lines = 500
x = np.linspace(0, 100, npoints) + np.random.normal(0, 2)
y = np.random.choice(range(100), size=len(x)).astype(float)
x = x.reshape((npoints, 1))
y = y.reshape((npoints, 1))
nodes = np.concatenate((x, y), axis=1)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
fig.patch.set_facecolor('black')

y_displacement = 50
y_offset = 0.1

for i in tqdm(range(n_lines)):
    x = nodes[:, 0]
    y = nodes[:, 1]

    x = x + np.random.normal(0, .5)
    y = y + i * y_offset + np.abs(np.random.normal(0, y_displacement, size=len(y)))

    tck, u = interpolate.splprep([x, y], s=0)
    xnew, ynew = interpolate.splev(np.linspace(0, 1, 100), tck, der=0)

    # set background to black
    ax.plot(xnew, ynew, color="white", linewidth=0.1)

ax.set_xlim(0, 100)
ax.axis("off")
plt.show()
