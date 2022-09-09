from unittest import TestCase

import numpy as np

from src.kohonen_som import KohonenSOM


class TestKohonenSom(TestCase):
    """Test class to test all functions of the KohonenSOM class."""

    def setUp(self) -> None:
        """Initialization of Class object."""
        # Initialize KohonenSOM class objects
        self.som = KohonenSOM(1, 1, 3)
        self.som2 = KohonenSOM(1, 1, 3)
        # Initialize additional KohonenSOM grid for testing
        self.test_som_grid = np.array([[[1, 1, 1], [0, 0, 0]]])

    def test_init(self) -> None:
        """Test if initialization is working correctly."""
        # Test if shape is applied correctly
        assert self.som.som_grid.shape[0] == 1
        assert self.som.som_grid.shape[1] == 1
        assert self.som.som_grid.shape[2] == 3
        # Test if radius and learning rate are applied correctly
        assert self.som.radius_0 == 0.5
        assert self.som.learning_rate_0 == 0.1
        # Test if randomness is working correctly
        assert not self.som == self.som2

    def test_sq_euclidean_distance(self) -> None:
        """Test caclulcation of squared euclidian distance."""
        vector_a = np.array([1, 1, 1], ndmin=2)
        vector_b = np.array([0, 0, 0], ndmin=2)
        distance = self.som._sq_euclidean_distance(
            vector_a=vector_a, vector_b=vector_b, axis=1
        )
        assert distance == 3

    def test_decay_function(self) -> None:
        """Test calculation of decay function used for radius / learning rate calculation."""
        value_0 = 1
        constant = 1
        decay_val = self.som._decay_function(
            value_0=value_0, constant=constant, epoch=1
        )
        assert not decay_val == 0

    def test_calculate_influence(self) -> None:
        """Test calculation of influence."""
        sq_distance = 1
        radius = 1
        influence = self.som._calculate_influence(
            sq_distance=sq_distance, radius=radius
        )
        assert not influence == 0

    def test_update_weights(self) -> None:
        """Test updating of weights."""
        vector_a = np.array([1, 1, 1])
        radius = 1.0
        learning_rate = 0.1
        bmu = (0, 0)
        assert np.array_equal(self.som.som_grid_initial, self.som.som_grid)
        self.som._update_weights(
            data_point=vector_a, radius=radius, learning_rate=learning_rate, bmu=bmu
        )
        assert not np.array_equal(self.som.som_grid_initial, self.som.som_grid)

    def test_train_som(self) -> None:
        """Test training of SOM."""
        data = np.array([1, 1, 1])
        epochs = 1
        self.som.train_som(train_data=data, epochs=epochs)
        assert not np.array_equal(self.som.som_grid_initial, self.som.som_grid)

    def test_find_bmu(self) -> None:
        """Test finding of BMU for datapoint."""
        vector_a = np.array([1, 1, 1])
        self.som.som_grid = self.test_som_grid
        bmu = self.som._find_bmu(current_input_vector=vector_a)
        assert bmu == (0, 0)
