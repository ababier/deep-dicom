import random
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class RotateAndFlip:
    """
    Randomly rotate (by 90° increments) and flip an image. Assumes the image is a numpy array with spatial dimensions on axes 2 and 3.
    """

    def __init__(self) -> None:
        self.rotation_options = [0, 1, 2, 3]  # Number of 90° rotations.
        self.flip_options = [None, "x", "y", "xy"]  # Flip options.
        self._num_90_degree_rotations: int = 0
        self._flip: Optional[str] = None

    def set_transform(self) -> None:
        """Randomly choose rotation and flip parameters."""
        self._num_90_degree_rotations = random.choice(self.rotation_options)
        self._flip = random.choice(self.flip_options)

    def __call__(self, image: NDArray) -> NDArray:
        """
        Apply the selected rotation and flip to the image.

        Args:
            image: Numpy array with spatial dimensions on axes 2 and 3.

        Returns:
            Transformed image array.
        """
        # Apply rotation.
        image = np.rot90(image, k=self._num_90_degree_rotations, axes=(2, 3))

        # Apply flip.
        if self._flip == "x":
            image = image[:, :, :, ::-1]
        elif self._flip == "y":
            image = image[:, :, ::-1, :]
        elif self._flip == "xy":
            image = image[:, :, ::-1, ::-1]

        # Ensure the array is contiguous.
        if np.any(np.array(image.strides) < 0):
            image = image.copy()
        return image
