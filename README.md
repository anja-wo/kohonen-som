# Kohonen SOM

Unsupervised learning model, Kohonen network, known as a Self-Organizing Map (SOM). Given a data set and it has to learn how to classify its components on its own, withouth labels given.

Dataset can consist of hundreds of rows and columns. Data is processed into a simplified map which provides a 2-dimensional representation of the training data set. This makes it much easier to read.

A Color example is used to train the SOM and to demonstrate its performance and typical usage.

:construction: WORK IN PROGRESS

## Tests

### Unit tests
All unit tests can be found in `test/`.

To execute the unit tests run
```
make test
```

## Code Quality and Style
To auto format the code styling with isort and black run
```
make style
```
To perform a static code quality chech run
```
make quality
```

## Set Up

For local training you might want to set up the project on your local machine:

First clone the code base and setup a Python virtual environment
1. Clone the code an ``cd`` in the directory
2. Create a Python3 Virtual environment, e.g. ``python3 -m venv venv``
3. Activate the created environment: ``source venv/bin/activate``
4. Install the requirements via ``pip install -r requirements.txt``

Afterwards you can train the model by running

```
python -m src.train --height {HEIGHT} --width {WIDTH} --epochs {EPOCHS}
```

Additional parameters (--radius, --learning_rate) can be provided as well.

## Docker
The project comes as a configured Docker container which can be built and run form the provided Dockerfile. You can build the Docker image locally by running

```
docker build -t kohonen-som .
````

The image i.e. the training can be run calling

```
docker run -p 80:80 -it kohonen-som -e HEIGHT={HEIGHT} -e  WIDTH={WIDTH} -e EPOCHS={EPOCHS}
```

If no ENV variables are provided, default Height, Width and Epochs are 10, 10 and 100.

To save model + image of trained SOM in current directory run
```
docker run -p 80:80 -e HEIGHT=10 -e WIDTH=10 -e EPOCHS=1000 -v $(pwd)/model:/app/src/model/ kohonen-som
```

The saved model can be loaded via
```
with open('model/som.p', 'rb') as infile:
    som = pickle.load(infile)
```

## Authors
* **Anja Wolf**
