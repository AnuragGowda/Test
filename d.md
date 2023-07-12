## Summary
A Neural Network is a type of computational model that is central to many Deep Learning (a subset of Machine Learning) algorithms. Despite their ubiquity in Deep Learning applications, many view them as a black box. In this *and subsequent* blog posts I will attempt to demystify the inner mechanisms that allow Neural Networks to function. 

>[!info] Info
>
> Familiarity with Python, NumPy and Mathematics (Linear Algebra and Calculus) will be required to fully understand content in subsequent postings
 
# Usage
Neural networks are used by feeding in numerical input data and getting numerical output data as a result. We can simplify usage into 3 main steps:
1. Encode input data into a numerical format, usually an array of numbers
2. Feed data to model
3. Get output/*prediction* from model

As an example, consider using a Neural Network to classify images of Cats and Dogs. 
1. We begin with encoding the images into an array of pixel color values  
2. We then feed this array to the model, which performs computation to produce an output. 
3. The output, which we may call the model's *prediction*, will be a numerical value, 0 if the model thinks the image is a dog and 1 if the model thinks the image is a cat. 

>[!example]+ Other Example Use Cases
>
> Some of the tasks Neural Networks can be used for include the following:
>- Facial Recognition (Computer Vision)
>- Cancer Detection
>- Stock Market Prediction
>- Text and Speech Recognition (Natural Language Processing)

# Definition
A Neural Network can be thought of as collections of neurons that are linked together. The firing of a specific neuron can affect the firing of other neurons, somewhat similar to the behavior of neurons in the brain. In fact, neural networks were originally (late 20th century) devised to closely mimic how the brain works--though the field of machine learning has recently begun progressing away from this idea.

## What is a Neuron?
A neuron can be thought of simply as an object that 
1. Takes in inputs from all connected neurons
2. Performs some simple computation on these values to produce a new value $z$
3. Outputs a modified version of $z$, called $a$ to all connected neurons

<center><figure class="image"><img src="https://lh3.googleusercontent.com/drive-viewer/AFGJ81qNqrXDeezw0jytEomeTF3RJ0NdE0ULZf_yD1fr0LPUpl5DYqulboQGuB_6Oh5T5Wh06bbMVvbe9t7VpPiVL_bA30EQJQ=s2560" alt="NN_Layers"><figcaption>Sample Neuron *f*</figcaption></figure></center>

The connections between these neurons are called weights and these can either be
<font color="#90EE90">excitatory</font> (i.e. an excitatory weight will amplify the value of $a$ output from a neuron) or <font color="#F4364C">inhibitory</font> (i.e. reducing the value of $a$ output). 

### Mathematically representing a neuron

#### Defining $z$
Typically, we mathematically represent this $z$ value as follows.  
 $$z_k = b_k +\sum_{i} w_ia_i$$

This equation tells us that the $z$ value of an arbitrary neuron $k$ is equal to the $a$ value all connected neurons are outputting times the weight $w$ that connects those neurons to our $k$th neuron. We also add an additional term $b_k$, which is called the *bias* term.

>[!question]- Why do we add a Bias?
>
>Adding a bias adds complexity and flexibility to a Neural Network, allowing it to learn quicker. This will be explored in subsequent posts.

#### Modifying $z$ to produce $a$
$z$ is then fed to an **activation function**, which determines the output value $a$ of the neuron, called the *activation* of the neuron. 

##### ReLU activation function
The most common activation function used today is ReLU (Rectified Linear Unit):
<center><figure class="image"><img src="https://sebastianraschka.com/images/faq/relu-derivative/relu_3.png" alt="NN_Layers"><figcaption>ReLU Graph - Source: <a href=https://sebastianraschka.com/faq/docs/relu-derivative.html>sebastianraschka.com</a></figcaption></figure></center>

The function can be denoted as $$\text{ReLU}(x) = \text{max}(0, x)$$

##### Defining $a$
It follows that we can define $a$ as  
$$a = \text{ReLU}(z)$$
which gives us the output value of a single neuron

## Putting Everything Together
Therefore, we can mathematically represent the activation of a neuron $a_k$ as follows:
$$z_k= b_k +\sum_{i} w_ia_i$$
$$a_k = \text{ReLU}(z_k)$$

>[!info]+ Info
>
>As you may notice, $Z$ is only an intermediary value, and it is not necessary to compute it to find the activation of a neuron. However, as we will see, it becomes helpful to keep this intermediary value for subsequent computations


## Neural Network Structure
Neural networks are composed of layers of neurons. In the case we will be focusing on, the layers are sequential in nature, meaning outputs from neurons in layer 1 are only passed to layer 2, outputs from neurons in layer 2 are only passed to neurons in layer 3 and so on. Typically, neural networks are dense. This means that all neurons in a previous layer will connect to all neurons in the subsequent layer (e.g. *each* neuron in layer 1 has a weight connecting it to *every* neuron in layer 2). 

The first layer is called the input layer and the last layer is the output layer. All other layers in-between do not have any interaction with the user and are therefore known as hidden layers.

<center><figure class="image"><img src="https://www.tibco.com/sites/tibco/files/media_entity/2021-05/neutral-network-diagram.svg" alt="NN_Layers"><figcaption>Neural Network Layers Example - Source: <a href=https://www.tibco.com/reference-center/what-is-a-neural-network>tibco.com</a></figcaption></figure></center>


But, how does this work in practice? And how does the model learn?

In the following posts, we explore these questions by building our own Neural Network from scratch, and teaching it how to recognize handwritten digits.
