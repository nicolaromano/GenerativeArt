from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

class FlowField:
    """
    A class to represent a flow field
    """

    def __init__(self, width: int = 100, height: int = 100, resolution: float = 0.1,
                neighbourhood_size: int = 3, decay: str = "inv_linear") -> None:
        """Initialize the flow field

        Args:
            width (int): the width of the field, defaults to 100
            height (int): the height of the field, defaults to 100
            resolution (float): the resolution of the field, defaults to 0.1
            neighbourhood_size (int): the size of the neighbourhood to use, defaults to 3. Particles are
                influenced by the neighbourhood_size x neighbourhood_size neighbourhood of vectors around them.
            decay (str): the decay function to use for the vector influence, defaults to "linear". 
                Can be one of "inv_linear", "inv_quadratic" or "inv_cubic".
        """

        self.width = width
        self.height = height
        self.resolution = resolution
        # Neighbourhood size must be > 1
        if neighbourhood_size < 2:
            print("Neighbourhood size must be > 1, setting to 2")
            neighbourhood_size = 2        
        self.neighbourhood_size = neighbourhood_size

        # Ensure the decay function is valid
        if decay not in {"inv_linear", "inv_quadratic", "inv_cubic"}:
            print("Invalid decay function, setting to linear")
            decay = "linear"
        self.decay = decay

        self.field = np.zeros((ceil(width / resolution), ceil(height / resolution), 2))
        self.init_field()

    def init_field(self, field_fn: callable = None) -> None:
        """Initialise the field with a given function
            
            Args:
                field_fn (callable): the function to initialise the field with. Defaults to None (in which case get_vector is used).
        """
        
        if field_fn is None:
            field_fn = self.__gen_vector

        # Create a grid of x and y coordinates
        x, y = np.meshgrid(np.arange(0, self.width, self.resolution),
                           np.arange(0, self.height, self.resolution))

        # Get the vector at each x,y coordinate
        field = np.array([field_fn(x, y) for (x, y) in zip(x.flatten(), y.flatten())])
        # Reshape the field to match the grid
        self.field = field.reshape(self.field.shape)

    def __gen_vector(self, x: float, y: float) -> np.ndarray:
        """
        Generates a flow-

        Args:
            x (float): the x coordinate
            y (float): the y coordinate

        Returns:
            np.ndarray: the vector at the given coordinates
        """

        x = 2 * np.pi * np.sin(x) + y
        y = 2 * np.pi * np.cos(y) + x

        # Check if the coordinates are within the field, otherwise loop around
        if x < 0 or x >= self.width:
            x %= self.width
        if y < 0 or y >= self.height:
            y %= self.height

        return [x, y] 

    def draw_field(self, color: str = 'black') -> None:
        """Draws the vector field

        Args:
            color (str, optional): The color of the vectors. Defaults to 'black'.
        """
        x, y = np.meshgrid(np.arange(0, self.width, self.resolution),
                           np.arange(0, self.height, self.resolution))

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.quiver(x.flatten(), y.flatten(), self.field[:, :, 0], self.field[:, :, 1], color=color)
        ax.axis('equal')
        ax.axis('off')
        plt.show()

    def get_vector(self, x: float, y: float) -> np.ndarray:
        """
        Get the compount vector at the given coordinates. We sum the neighborhood of 
        the given coordinates weighted by the decay function.

        Args:
            x (float): the x coordinate
            y (float): the y coordinate

        Returns:
            np.ndarray: the vector at the given coordinates
        """

        # Find the neighbourhood of the given coordinates
        x_min = int(x - self.neighbourhood_size / 2)
        x_max = int(x + self.neighbourhood_size / 2)
        y_min = int(y - self.neighbourhood_size / 2)
        y_max = int(y + self.neighbourhood_size / 2)

        # Check if the coordinates are within the field, otherwise loop around
        if x_min < 0 or x_max >= self.width:
            x_min %= self.width
            x_max %= self.width
        if y_min < 0 or y_max >= self.height:
            y_min %= self.height
            y_max %= self.height

        # Get the neighbourhood of vectors
        vectors = self.field[x_min:x_max, y_min:y_max, :]
        # Get the neighbourhood of x and y coordinates
        x_coords = np.arange(x_min, x_max) * self.resolution
        y_coords = np.arange(y_min, y_max) * self.resolution

        # Get the distance of each vector from the given coordinates
        distances = np.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2)
        # Get the weights for each vector
        weights = self.__get_weights(distances)

        # Sum the vectors weighted by the weights
        return np.sum(vectors * weights, axis=(0, 1))

    def __get_weights(self, distances: np.ndarray) -> np.ndarray:
        """
        Get the weights for the given distances

        Args:
            distances (np.ndarray): the distances to get the weights for

        Returns:
            np.ndarray: the weights for the given distances
        """

        if self.decay == "inv_linear":
            return 1 / distances
        elif self.decay == "inv_quadratic":
            return 1 / distances ** 2
        elif self.decay == "inv_cubic":
            return 1 / distances ** 3

    def __str__(self) -> str:
        """_summary_ of the flow field

        Returns:
            str: summary of the flow field
        """
        return f"FlowField: {self.cols} x {self.rows} @ {self.resolution}"

class Particle:
    """
    A class to represent a particle. Particles get moved around by the flow field.
    """

    def __init__(self, x:float = 0.0, y:float = 0.0, lifespan:int = 100, color:str = "black") -> None:
        """
        Initialize a particle.

        Args:
            x (float, optional): The x position. Defaults to 0.0.
            y (float, optional): The y position. Defaults to 0.0.
            lifespan (int, optional): The lifespan of the particle. Defaults to 100.
            color (str, optional): The colour of the particle. Defaults to "black".
        """
        self.x = x
        self.y = y
        self.lifespan = lifespan
        self.color = color
        self.pos = np.array([x, y])
        
    def flow_particle(self, field:FlowField) -> None:
        """
        Move the particle according to the flow field.
        The particle is influenced by the vectors around it, according to its distance.

        Args:
            field (FlowField): The flow field to use.
            n_steps (int, optional): The number of steps to take. Defaults to 100.
        """

        for _ in range(self.lifespan):
            # Get the sum of the vectors in the neighbourhood around the particle
            vector = field.get_vector(self.x, self.y)
            # Move the particle according to the vector and append the new position to the list            
            self.x += vector[0]
            self.y += vector[1]
            self.pos = np.vstack((self.pos, np.array([self.x, self.y])))            


ff = FlowField(1, 1, 0.05)
#ff.draw_field()
n_particles = 3
x = np.random.randint(0, 10, n_particles)
y = np.random.randint(0, 10, n_particles)
particles = [Particle(x[i], y[i], lifespan=100) for i in range(n_particles)]

for p in particles:
    p.flow_particle(ff)
    print(p.pos)
