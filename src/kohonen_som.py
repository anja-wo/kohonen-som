#!/usr/bin/env python3

import copy
from math import log
from typing import Any
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


class KohonenSOM:

    """
    Class to intialize a Kohonen Network with given parameters width, height and input vector size.
    Includes all functions to train the initialized network with a given training data set.
    """

    def __init__(
        self,
        som_width: int,
        som_height: int,
        input_vec_size: int,
        radius_0: Optional[float] = None,
        learning_rate_0: Optional[float] = None,
    ) -> None:
        """Init method

        Parameters
        ----------
        som_width : int
            Width of Kohonen SOM
        som_height : int
            Height of Kohonen SOM
        input_vec_size : int
            Size of input vector
        radius_0 : Optional[float], optional
            Radius for training start
        learning_rate_0 : Optional[float], optional
            Learning rate for training start
        """
        # Initiate SOM grid with passed input variables
        self.som_grid_initial = np.random.random(
            (som_width, som_height, input_vec_size)
        )
        self.som_grid = copy.deepcopy(self.som_grid_initial)

        if radius_0:
            if radius_0 >= som_height or radius_0 >= som_width:
                print(
                    "Warning: Radius is too high for the dimension of the map, default value is used"
                )
                # Set default radius
                self.radius_0 = max(som_height, som_width) / 2
            else:
                self.radius_0 = radius_0
        else:
            # Set default radius
            self.radius_0 = max(som_height, som_width) / 2

        if learning_rate_0:
            self.learning_rate_0 = learning_rate_0
        else:
            # Set default learning rate
            self.learning_rate_0 = 0.1

    def train_som(self, train_data: np.ndarray, epochs: int) -> None:
        """
        Function to train the initialized network with the training data given
        and for the amount of epochs stated.

        Parameters
        ----------
        train_data : np.ndarray
            Data to train SOM with
        epochs : int
            Epochs to run training for
        """

        rand = np.random.RandomState(0)

        # Set current values to initial values
        radius = self.radius_0
        learning_rate = self.learning_rate_0

        # Calculate time constant
        time_constant_r = epochs / log(self.radius_0)

        # Iterate for defined number of epochs
        for epoch in range(1, epochs + 1):
            # print(f"Running epoch {epoch}: Radius {radius}; Learning rate: {learning_rate}")
            rand.shuffle(train_data)
            # Choose one random data point in the training data set and find bmu / update weights
            current_input_vector = train_data[0]
            bmu = self._find_bmu(current_input_vector)
            self._update_weights(current_input_vector, radius, learning_rate, bmu)

            # Update radius and learn rate
            radius = self._decay_function(self.radius_0, time_constant_r, epoch)
            learning_rate = self._decay_function(
                self.learning_rate_0, time_constant_r, epoch
            )

    def _sq_euclidean_distance(
        self, vector_a: np.ndarray, vector_b: np.ndarray, axis: int
    ) -> Any:
        """
        Function to calculate the squared Euclidean Distance.
        No Square root caluclated to improve runtime.

        Parameters
        ----------
        vector_a : np.ndarray
            First vector to calculate distance from
        vector_b : np.ndarray
            Second vector to calculate distance to
        axis : int
            Over which axis to build sum

        Returns
        -------
        Any
            Calculated distance, either as array or single number
        """
        return (np.square(vector_a - vector_b)).sum(axis=axis)

    def _decay_function(self, value_0: float, constant: float, epoch: int) -> float:
        """
        Function to calculate radius and learning rate based on epoch and constant.

        Parameters
        ----------
        value_0 : float
            Value to start with
        constant : float
            Constant
        epoch : int
            Current epoch number

        Returns
        -------
        float
            Calculated constant value
        """
        return value_0 * np.exp(-epoch / constant)

    def _calculate_influence(self, sq_distance: float, radius: float) -> float:
        """
        Function to caluclate influence of neighbouring nodes of BMU based on squared euclidean distance and radius.

        Parameters
        ----------
        sq_distance : float
            Squared Euclidean distance of neigbouring node
        radius : float
            Current used radius

        Returns
        -------
        float
            Calculuated influence for neigbouring node
        """

        return np.exp(-sq_distance / (2 * (radius ** 2)))

    def _find_bmu(self, current_input_vector: np.ndarray) -> Tuple[int, int]:
        """
        Function to find the Best Matching Unit (BMU) using the squared Euclidean distance.

        Parameters
        ----------
        current_input_vector : np.ndarray
            Data point to find BMU for

        Returns
        -------
        Tuple[int, int]
            BMU as matrix value, e.g. (1,1)
        """

        distances = self._sq_euclidean_distance(current_input_vector, self.som_grid, 2)
        minimum = np.argmin(distances, axis=None)
        bmu = np.unravel_index(minimum, distances.shape)

        return bmu

    def _update_weights(
        self,
        data_point: np.ndarray,
        radius: float,
        learning_rate: float,
        bmu: Tuple[int, int],
    ) -> None:
        """
        Function to update weights of SOM grid based on datapoint, its bmu, radius, learning rate

        Parameters
        ----------
        data_point : np.ndarray
            Current datapoint on which update weights is based on
        radius : float
            Current radius
        learning_rate : float
            Current learning rate
        bmu : Tuple[int, int]
            BMU of current data point
        """

        # Get x, y value for Best Matching Unit
        x, y = bmu

        # Iterate through SOM to update weigts for all neighbouring nodes within radius distance
        for i in range(0, self.som_grid.shape[0]):
            for j in range(0, self.som_grid.shape[1]):
                # Calculate square distance from x/y values of BMU and Node and build sum
                sq_distance = np.square(i - x) + np.square(j - y)
                # sq_distance = self.sq_euclidean_distance(np.array([i, j]), np.array([x, y]), 0)
                # sq_distance = np.sum((np.array([i, j]) - np.array([x, y])) ** 2)
                if sq_distance <= radius ** 2:
                    # If within calculated neighbourhood, adapt weights
                    influence = self._calculate_influence(sq_distance, radius)
                    self.som_grid[i, j, :] += (
                        learning_rate
                        * influence
                        * (data_point - self.som_grid[i, j, :])
                    )

    def plot_results(self, train_data: np.ndarray, input_vec_size: int) -> None:
        """Saving results as an image."""

        new_size = int(train_data.size / input_vec_size)

        fig, ax = plt.subplots(
            nrows=1, ncols=3, figsize=(12, 3.5), subplot_kw=dict(xticks=[], yticks=[])
        )
        ax[0].imshow(train_data.reshape(1, new_size, input_vec_size), vmin=0, vmax=1)
        ax[0].title.set_text("Train Data")
        ax[1].imshow(self.som_grid_initial, vmin=0, vmax=1)
        ax[1].title.set_text("Som Initial Data")
        ax[2].imshow(self.som_grid, vmin=0, vmax=1)
        ax[2].title.set_text("Som trained")

        plt.savefig("./src/model/result.png")
