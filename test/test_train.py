import sys
from unittest import TestCase
from unittest import mock

from src.train import run_train


class TestTrain(TestCase):
    """Test class to E2E test training of KohonenSOM."""

    def test_train_missing_paramters(self) -> None:
        """Function to test training with missing parameters."""
        with mock.patch("sys.argv", ["module", "--height", "10", "--width", "10"]):
            with self.assertRaises(TypeError):
                run_train(sys.argv)

    def test_train_no_radius_learning_rate(self) -> None:
        """Function to test training without radius / learning rate given."""
        with mock.patch(
            "sys.argv", ["module", "--height", "10", "--width", "10", "--epochs", "10"]
        ):
            run_train(sys.argv)

    def test_train_radius(self) -> None:
        """Function to test training with radius / learning rate given."""
        with mock.patch(
            "sys.argv",
            [
                "module",
                "--height",
                "10",
                "--width",
                "10",
                "--epochs",
                "10",
                "--radius",
                "10",
            ],
        ):
            run_train(sys.argv)

    def test_train_radius_learning_rate(self) -> None:
        """Function to test training with radius / learning rate given."""
        with mock.patch(
            "sys.argv",
            [
                "module",
                "--height",
                "10",
                "--width",
                "10",
                "--epochs",
                "10",
                "--radius",
                "10",
                "--learning_rate",
                "0.5",
            ],
        ):
            run_train(sys.argv)

    def test_train_data(self) -> None:
        """Function to test training without radius / learning rate but data given."""
        with mock.patch(
            "sys.argv",
            [
                "module",
                "--height",
                "10",
                "--width",
                "10",
                "--epochs",
                "10",
                "--data",
                "/fake/path",
            ],
        ):
            with self.assertRaises(RuntimeError):
                run_train(sys.argv)
