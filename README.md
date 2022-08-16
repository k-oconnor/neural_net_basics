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
