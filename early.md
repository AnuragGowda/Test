---
creation date: Wednesday July 12th 2023 14:34:54
modification date: Wednesday July 12th 2023 14:34:54 
---
# MNIST - Recognizing handwritten digits
A classic introductory way to learn Machine Learning is to create a model that is able to recognize handwritten digits. Many use the MNIST (Modified National Institute of Standards and Technology) database, which contains thousands of examples of labeled handwritten digits. In this series, we will take this a step further and implement a Neural Network without using either TensorFlow/Keras or PyTorch (popular machine learning libraries), just Python and NumPy.

## Problem Introduction
The goal is to create a Neural Network model where we can input an image of a handwritten digit and get an output of what digit our input was. 

### Input and Output Specification
More precisely, the input will consist of a greyscale $28\times28$ pixel image, represented as an array of pixel intensities (i.e. $0= \text{black}$ , $255 = \text{white}$). The output then will be an array of length ten, where every index will have a percentage which is how likely the model thinks the example is that particular number. We can then pick the index with the highest probability as the model's overall prediction.

INSERT IMAGE

### Network Architecture 
Since the input is a $28 \times 28$ pixel grid, this is equivalent to a one-dimensional array of size $784$. Therefore, our input layer will have $784$ neurons/units. We will have one hidden layer with $20$ units, and since the output is a one-dimensional array of size 10, the output layer will have $10$ units.
  

$$
A^{(0)} \rightarrow A^{(1)} \rightarrow A^{(2)}
$$
We may also call the input layer "layer 0", the hidden layer "layer 1", and the output layer "layer 2".

>[!question]- Why do we only have 20 units in layer 1 and only 3 total layers?
>
> Generally, the more units/layers we have the better our model will be. However, since we are creating the model from scratch, we want to keep the size small to aid ease of future math and development. From there, we can later begin to expand the model.

# Coding Setup
In this section, we will look at the data we have and format and analyze it.

## Obtaining Data

Use this link to get the .csv file for our data. 

>[!Warning]-
>
>Make sure to keep the file in the same folder as the Python file we will be using to make our Neural Network

## Data Setup

>[!info]- 
>
> For the following steps, make sure you have Numpy and Python installed. You may want to use a Jupyter Notebook to put all the code into, but putting everything into a single Python file will work fine as well.

Lets import NumPy and split our data into two sets. With one set, we can train our model, and with the other, we can verify the accuracy of the model. There are 42,000 total samples, so we will use 30,000 for training and the rest for testing.

```python
import numpy as np

data = np.genfromtxt("train.csv", delimiter=",")[1:]

train, test = np.split(data, [30000])
y_train, x_train = np.split(train.T, [1])
y_test, x_test = np.split(train.T, [1])
```

>[!Success]
>
> Try printing out values of `x_train` and `y_train` to see what they look like

Lets also initialize the weights and biases. It's good practice to allow the weights to be 
somewhat random, allowing for our model to more easily *converge*.

```python
W1 = np.random.rand(20, 784) - 0.5
B1 = np.zeros((20, 1))
W2 = np.random.rand(10, 20) - 0.5
B2 = np.zeros((10, 1))
```

>[!Question]- What is convergence?
>
> Convergence has to do with a central idea behind *Gradient Descent*, an algorithm we will explore later, but for now, convergence can be thought of as the idea of the model reaching a point where it is able to make high accuracy predictions.

## Data Representation
All our variables are storing data as two-dimensional matrices/arrays. For example, B2 is a zero matrix of shape (10, 1), meaning that it has 10 rows and 1 column, and looks like the following:

Here are the shapes of the other variables:
- `x_train`: shape (30000, 784), each row contains the pixel data for an image of one handwritten digit
- `y_train`: shape (1, 30000), since this only has one row, it can be thought of as a normal array with each $i^{th}$ index being the value of the corresponding digit data in the $i^{th}$ row of `x_train`
- `x_test`: shape (12000, 784), same as `x_train`, each row is data for one image
- `y_test`: shape (1, 12000), same as `y_train`

> [!tip]- Visualizing images
> 
> If you want to see what a particular image in `x_train` looks like, you can use 
<br>
- `W1`: shape (20, 784), with $j=20$ rows corresponding to the amount of units in the first layer and $k=784$ columns corresponding to the units in the input layer.  Each entry in this matrix is therefore a weight spanning from a $j^{th}$ unit in the hidden layer to a $k^{th}$ unit in the input layer
- `B1`: shape (20, 1), each row represents a neuron in the first layer
- `W2`: shape (10, 20), weights from hidden layer to the output layer
- `B2`: shape (10, 1), each row is a neuron in the output layer

> [!question]- Why is Understanding Data Shape Important?
> 
> To begin to understand how our model makes predictions, it is important to understand how the architecture of the model is set up
