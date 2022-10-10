import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from collections.abc import Callable
from typing import Any


class PointGrid:
    """A class to represent a grid of points, that can be transformed in various ways.
    """

    def __init__(self, width=100, height=100):
        coords = [[x, y] for x in range(width) for y in range(height)]
        self.height = height
        self.width = width
        self.grid = np.array(coords, dtype=float)
        self.colour = ["gray" for x in range(width) for y in range(height)]
        self.ptsize = [1 for x in range(width) for y in range(height)]

    def __str__(self):
        return f"A PointGrid object of {self.height} x {self.width} points."

    def ident(self, x: np.array) -> np.array:
        """The identity function

        Args:
            x (np.array): A set of coordinates

        Returns:
            np.array: The same coordinates
        """
        return x

    def transform(self, func: Callable[..., np.array] = ident, *args) -> None:
        """Transforms the grid points

        Args:
            func (Callable[..., np.array], optional): The function to apply to the x coordinates of the grid points. The function takes at least one single argument containing the coordinates and returns the transformed coordinates. Defaults to the identity function.
            *args: any additional arguments to pass to func
        """
        res = self.grid.copy()

        for i, pt in enumerate(self.grid):
            res[i] = func(pt, *args)

        self.grid = res

    def map_color(self, func: Callable, *args) -> None:
        """Sets the colour of each point according to a function

        Args:
            func (Callable): A function to apply to each point, which will return a colour. The function takes at least one single argument containing the coordinates and returns the colour.
            *args: any additional arguments to pass to func
        """
        for i in range(len(self.colour)):
            self.colour[i] = func(self.grid[i], *args)

    def map_size(self, func: Callable[..., float], *args) -> None:
        """Sets the size of each point according to a function

        Args:
            func (Callable): A function to apply to each point, which will return a size. The function takes at least one single argument containing the coordinates and returns the colour.
            *args: any additional arguments to pass to func
        """
        for i in range(len(self.ptsize)):
            self.ptsize[i] = func(self.grid[i], *args)

    def plot(self, alpha=0.5, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(x=self.grid[:, 0], y=self.grid[:, 1],
                   s=self.ptsize, alpha=alpha, color=self.colour)
        ax.axis("off")


def cart2pol(coords):
    x, y = coords
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return([rho, phi])


def pol2cart(coords):
    rho, theta = coords
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return([x, y])


def jitter(coords: np.array, amount: float) -> np.array:
    return(coords + np.random.normal(scale=amount, size=2))


def scale(coords: np.array, amount: float) -> np.array:
    return(coords * amount)


def transform1(coords: np.array, a: float, b: float, c: float, d: float) -> np.array:
    x, y = coords
    x = a * np.sin(x) + b * np.cos(y) + c * np.sin(x / y)
    y = a * np.cos(x) + d * np.sin(y)
    coords = [x, y]
    return(coords)


def color_fun(coords: np.array, a: float, b: float, c: float, d: float):
    coords = coords / 4
    r = 0.5 * (np.sin(a*coords[1]) + 1)
    g = 0.5 * (np.cos(b*coords[0]) + 1)
    b = 0.5 * (np.sin(c*coords[0] - d*coords[1]) + 1)
    return ((r, g, b))


def size_fun(coords: np.array):
    return coords[0] * coords[1] + .4


tr_params = [[a, 2, 1, .4] for a in np.linspace(0.2, -0.3, 6)]

fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(18, 3))

for i, axis in enumerate(ax):
    pg = PointGrid(150, 150)
    # pg.map_color(color_fun, 20, 5, 8, 5)
    a, b, c, d = tr_params[i]
    pg.transform(transform1, a, b, c, d)
    pg.map_size(size_fun)

    pg.plot(alpha=0.4, ax=axis)

plt.show()
