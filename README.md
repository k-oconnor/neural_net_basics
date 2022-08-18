# Building a Neural Network with PyTorch
## _Machine Learning from Disaster Dataset_
### DATA: https://www.kaggle.com/competitions/titanic/overview

## What is a neural network (nn) ?
- A neural network is a method in artificial intelligence that teaches computers to process data in a way that is inspired by the human brain. 
- It is a type of machine learning process, called deep learning, that uses interconnected nodes or neurons in a layered structure that resembles the human brain. 
- It creates an adaptive system that computers use to learn from their mistakes and improve continuously.

![A 1-Hidden Layer NN with two neurons](Images/Fig1.png)

## What are the building blocks of a nn in PyTorch?
- Tensors: A tensor is a matrix-like data structure that is used across deep learning frameworks. Tensor operations are fundamental to PyTorch, and are used for all data.
- Dataset class constructor: The dataset constructor at a minumum, has to have two methods created. __len__ , and __getitem__ to apply to the tensors in the dataset. 
    it is also commonly where input dataframes are converted to tensors, and normalization transforms are applied.
- Input layer: The input layer is said to be x-dimensional, where x is the number of explanatory variables or features.
- Hidden Layer(s): The hidden layers are constructed of artificial neurons, which are sequential applications of linear functions which are passed through activation functions.
- Output Layer: In the case of binary classification, the output layer consists of a consolidating linear function, which is passed through a sigmoid function to output a probability for class. In the binary case, the output is of dimension 1. In multi-class classification, the output is n-dimensional, where n is the number of possible class.
- Loss Function: The loss function is the criterion on which the network measures success. For example, in simple linear regression, we try to minimize mean square error. There are numerous loss funcitons that can be used for different applications. Binary cross-entropy loss is a commonly used for single-class prediction. Cross-entropy loss can be extended for use in multi-class classification.
- Optimizer: The optimizer is a function or algorithm that updates the network parameters at each epoch, and is integral to the learning process. The weights of the parameters are increased or decreased progressively in attempt to reduce loss on the next pass.

## When does learning become "deep"?
> The distinction between learning 
> and deep learning is in the number of 
> hidden layers used. If a model uses
> two or more hidden layers, we refer
> to the model as a deep learning model.

## Problem Description and Data
In April of 1912, one of the most infamous naval disasters in modern history occured, when the RMS Titanic struck an iceberg and began to sink. Tragically, there were not enough life boats for the majority of the passengers aboard. 1502 of the 2224 people aboard the ship lost thier lives in the ensuing chaos. 

Seemingly, there were factors that made it more likely for an individual to survive the disaster. For example, women and children were prioritized to get space on the life boats. Given data on socio-economic factors, can we predict who would have survived in some sort of systematic fashion?

That is a the goal of the competition. We are given two csv files with missing and messy data. The training set has records of 891 individuals and survival ground truth. The testing set has 418 records, and we are tasked with making a survival prediction for each individual.

## Features

While this readme is geared towards building a neural network from scratch, it is important to highlight that all feature engineering and data cleaning must be performed prior to building and training the model. Is is also important that any cleaning steps and engineering must be applied to training, validation, and testing data uniformly.

| Feature | Description |
| ------ | ------ |
| 'Survived' | Binary outcome of survival |
| 'PassengerID' | Unique identifier for each passenger |
| 'Pclass' | What class ticket did the individual have (1st,2nd,3rd)|
| 'Sex' | 'Male' or 'Female' records |
| 'Age' | Age in years|
| 'Sibsp' | # of siblings/spouses aboard |
| 'Parch' | # of parents/children aboard |
| 'Fare' | How much the ticket costed in 1912 GBP |
| 'Embarked' | Which port did the individual embark from? |
| 'Cabin' | If the individual had a cabin, what cabin was it? |
| 'Has_Cabin' | Did the individual have a cabin? |
| 'Title' | If the individual has a title preceding thier name; ranked by rarity |

From the table, the last two variables I engineered. 

'Has_Cabin' is a simple binary indicator (1 = has a cabin, 0 = no cabin)
!['Has_Cabin' Lambda Function](Images/code2.png)

'Title' was constructed by extracting the the characters at the start of each passengers name, and making a ranking scheme based on the rarity of the titles.
For example 'Mr.' is very common, where the title 'Lord' signifies upper class land owners.
![Title Extraction Code](Images/code1.png)

Please refer to the code comments for the steps I took with cleaning and missing value imputation for the rest of the variables.

# The Steps 
After completion of preprocessing, we can break down the process of model construction into the following typical steps:
- Dataset Constructor and Loading
- Model Constructor
- Selecting Optimizer and Loss Metric
- Training, Validation, and Tuning
- Making Predictions on Test Data

## Dataset Constructor and Loading

Code for processing data samples can get messy and hard to maintain; we ideally want our dataset code to be decoupled from our model training code for better readability and modularity. PyTorch provides two data primitives: torch.utils.data.DataLoader and torch.utils.data.Dataset that allow you to use pre-loaded datasets as well as your own data. Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples. The dataset class constructor requires at a minimum the methods " __getitem__" and "__len__". Apart from applying these methods, the constructor is also used for converting the data to tensors.

![Dataset Constructor](Images/code3.png)

Notice that we are also using SKLEARN to split our training data into testing and validation sets. The random seed can be any integer. It just ensures that results are reproducible, by maintaing the same starting position from a psuedo-random process. Test size specifies the proportion of the data we withhold for the model validation.

After splitting the data, we create the data primitives 'DataLoader' after applying the Class constructor to our datasets.

![Creating the data loaders](Images/code5.png)

## Model Constructor
The model constructor is where we make our specifications for the neural network. First, we make our linear layers. The forward method can be called to make a prediction. There are three typical activation functions 'relu, tanh, sigmoid'. Relu is useful, as sigmoid and tanh are bounded <|1|, so as multiple gradients are multiplied together, the number doesn't neccesarily approach 0. We apply the relu function in the hidden layers, and use a sigmoid activation in the output layer to make predictions. Passing the linear function through the sigmoid activation will result in output being the probability of survival, analogous to a logistic regression. Note how there are two hidden layers. We can generalize this construction to an arbitrary number of layers, by adding more linear functions and activations to the __init__ and the forward pass method. 

Reminder: A neuron is a combination of a linear function and an activation
- Input dimension is the number of explanatory variables
- H1 is the number of neurons in the first hidden layer
- H2 is the number of neurons in the second hidden layer
- Output dimesion is the number of classes we are predicting (In this case, it is binary)

![Model Specification](Images/code4.png)

Note how at the end of the constructor, we initialize the model with our desired specification. The number of neurons and hidden layers needs to be carefully tuned. Having too many neurons and layers can lead to extreme overfitting. Having too few will cause our paramter estimates to underfit. It is important to simultaneously train and validate to find the sweet spot. There is no perfect answer, and there is a lot of room for intuition. The model as specified takes the following form.

![Model Initialization](Images/Fig3.png)

## Optimizer and Loss Function
![Model Initialization](Images/code6.png)


## Training the Model
![Model Initialization](Images/cod7.png)

## Predicting Results