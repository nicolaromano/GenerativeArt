import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.patches import Rectangle
import numpy as np
import itertools

num_rows = 12
num_cols = 12
square_size = 5
gap_size = 0.1

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

def logit(p):
    return(1/(1+np.exp(-p)))

for r, c in itertools.product(range(num_rows), range(num_cols)):
    cx = c * (square_size + gap_size)
    cy = r * (square_size+gap_size)
    for i in range(square_size):
        red = logit((r+1)*(10*i+1)/square_size)
        green = logit((c+1)*(0.05*i+1)/square_size)
        blue = logit(((r+1)/(c+1))*(0.01*i+1)/square_size+np.random.normal(0, .4))
        rect = Rectangle((cx + i/2, cy + i/2), square_size - i, square_size-i,
                         fill=False, angle=c+r,
                         color=(red, green, blue))
        ax.add_patch(rect)

ax.set_xlim(-square_size-gap_size, (num_cols + 1) * (square_size + gap_size))
ax.set_ylim(-square_size-gap_size, (num_rows + 1) * (square_size + gap_size))
ax.axis("off")
plt.show()
