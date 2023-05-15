# Neuroflux deep Learning Application

This application uses deep learning to predict Neuroflux disorder phases based on MRI scans. It provides two models for this purpose, both defined in the `main.py` script.

## Table of Contents

- [Application](#application)
  - [main.py](#mainpy)
- [Packaging](#packaging)
  - [Dockerfile](#dockerfile)
  - [docker-compose.yml](#docker-composeyml)
- [Dockerization](#dockerization)
- [Conclusion](#conclusion)


## Application

### main.py

The `main.py` file contains the Python code for the deep learning models. It consists of three classes:

- `BaseModel`: This is the base class for the models. It defines the common functionalities like data loading, model evaluation, and prediction.
- `Model1`: This class inherits from `BaseModel` and implements the `train` method using a pre-trained ResNet50 model.
- `Model2`: This class also inherits from `BaseModel` but it implements the `train` method using a custom Convolutional Neural Network (CNN).
## Packaging
### Dockerfile
The Dockerfile defines the environment in which the application runs. It starts from a Python 3.8 image, installs the necessary dependencies from the requirements.txt file, copies the application files into the image, and finally runs the main.py script when a container is launched from the image.

### docker-compose.yml
The docker-compose.yml file is used to define and run the multi-container Docker application. It builds the Docker image using the Dockerfile, maps port 5000 of the container to port 5000 of the host, and mounts the data directory from the host to the data directory in the container.


## Dockerization
Dockerization makes it easy to create a reproducible environment for the application. The application and its dependencies are packaged into a Docker image, which can be run consistently on any platform that supports Docker.

The Docker image for this application is defined in the Dockerfile. To build the image, navigate to the project directory and run:
docker build -t neuroflux/app .

To run a container from the image, use:
docker run -p 5000:5000 neuroflux/app

To run the application using Docker Compose, use:
docker-compose up

don't forget to change the path of data in docker-compose.yml file




