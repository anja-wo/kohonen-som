#!/usr/bin/env python3

import argparse
import pickle
import sys
import time

import numpy as np

from src.kohonen_som import KohonenSOM


def run_train(args) -> None:  # type: ignore
    """
    Function to initialize a random SOM and train it with the data points provided

    Parameters
    ----------
    args: list
        List of arguments passed
    """

    parser = argparse.ArgumentParser(description="Parameters for Training")
    # Required arguments
    parser.add_argument("--width", dest="width", type=int)
    parser.add_argument("--height", dest="height", type=int)
    parser.add_argument("--epochs", dest="epochs", type=int)
    # Optional arguments
    parser.add_argument("--radius", dest="radius", type=int, nargs="?")
    parser.add_argument("--learning_rate", dest="lr", type=float, nargs="?")
    parser.add_argument("--data", dest="data", type=str, nargs="?", help="Path to data")

    args = parser.parse_args()

    width = args.width
    height = args.height
    epochs = args.epochs
    radius = args.radius
    learning_rate = args.lr

    if args.data:
        # TODO Read file + pre-process entries for training SOM
        raise RuntimeError("Currently no specified dataset can be used for training.")
    else:
        # Create random data set with 20 entries
        train_data = np.random.random((20, 3))

    print(f"--Create SOM of size {width} x {height} and train with {epochs} epochs--")

    start_time = time.time()

    # Set input vector size based on training data set
    input_vec_size = train_data.shape[-1]
    # Initiate KohonenSOM object with passed parameters
    som = KohonenSOM(width, height, input_vec_size, radius, learning_rate)
    # Train SOM
    som.train_som(train_data, epochs)
    end_time = time.time()

    print(f"The training took {(end_time - start_time)} seconds.")

    # Saving the som in the file som.p
    with open("src/model/som.p", "wb") as outfile:
        pickle.dump(som, outfile)

    som.plot_results(train_data, input_vec_size)


if __name__ == "__main__":
    run_train(sys.argv)
