from tracemalloc import stop
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from torch.nn import BCELoss,MSELoss
from torch.optim import SGD
from torch import sigmoid,tanh,relu
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
import csv


## Loading in our data
valid = pd.read_csv("test.csv")
base = pd.read_csv("train.csv")

## ---------------- data cleaning and feature engineering  ---------------- ##


## Mapping sex to binary categorical
base.loc[base.Sex != 'male', 'Sex'] = 1
base.loc[base.Sex == 'male', 'Sex'] = 0

## Creating a new binary variable "Has_Cabin", which is = 1, if the passenger has a cabin.
base['Has_Cabin'] = base["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

## We assume that if there is no cabin information, the passenger doesn't have a cabin. We replace all 'NA's with 0.
base.Cabin.fillna('0', inplace=True)

## In researching accomodations on the Titanic, we observe that decks A-C are exclusive to first class.
## D-deck is predominately 2nd class, E-deck is a mix of 2nd and 3rd, and F-T are mostly 3rd class.
## We want this factor to increase, as the passenger is housed near the top of the ship, and we assume that,
## lower classes are closer to the bottom of the ship, with the cheapest fares in community areas near storage and engines.

base.loc[base.Cabin.str[0] == 'A', 'Cabin'] = 4
base.loc[base.Cabin.str[0] == 'B', 'Cabin'] = 4
base.loc[base.Cabin.str[0] == 'C', 'Cabin'] = 4
base.loc[base.Cabin.str[0] == 'D', 'Cabin'] = 3
base.loc[base.Cabin.str[0] == 'E', 'Cabin'] = 2
base.loc[base.Cabin.str[0] == 'F', 'Cabin'] = 1
base.loc[base.Cabin.str[0] == 'G', 'Cabin'] = 1
base.loc[base.Cabin.str[0] == 'T', 'Cabin'] = 1
base['Cabin'] = base['Cabin'].astype(int)

## This is a basic exploratory plot that shows the distribution of passenger class across the cabin levels.
## Our assumptions above appear to be correct. The higher the passenger class, the higher up on the ship thier accomodations are.
fare_bar = plt.bar(base['Cabin'], height = base['Pclass'])
plt.show()

## Checking if there are any missing passenger classes
print(base['Pclass'].isna().sum())
print(valid['Pclass'].isna().sum())

## Checking if there are any missing fares, and grouping by passenger class to make a choice about filling NA values
print(base['Fare'].isna().groupby(base['Pclass']).sum())
print(valid['Fare'].isna().groupby(valid['Pclass']).sum())

## Encoding fares to a multi-level factor which increases with fare cost
base.Fare.fillna(0, inplace=True)
base.loc[(base['Fare'] <= 8), 'Fare'] = 0
base.loc[(base['Fare'] > 8) & (base['Fare'] <= 16), 'Fare'] = 1
base.loc[(base['Fare'] > 16) & (base['Fare'] <= 30), 'Fare'] = 2
base.loc[(base['Fare'] > 30) & (base['Fare'] <= 100), 'Fare'] = 3
base.loc[(base['Fare'] > 100), 'Fare'] = 4
base['Fare'] = base['Fare'].astype(int)

## A simple function for extracting titles from names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
            return title_search.group(1)
    return ""

## Retrieving titles and mapping them based on thier rarity and class implications
base['Title'] = base['Name'].apply(get_title)
    # Group all non-common titles into one single grouping "Rare"
mapping = {'Mlle': 'Rare', 'Major': 'Rare', 'Col': 'Rare', 'Sir': 'Rare', 'Rev': 'Mr',
               'Don': 'Rare', 'Mme': 'Rare', 'Jonkheer': 'Mr', 'Lady': 'Rare',
               'Capt': 'Mr', 'Countess': 'Rare', 'Ms': 'Miss', 'Dona': 'Rare'}
base.replace({'Title': mapping}, inplace=True)
    # Mapping titles
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
base['Title'] = base['Title'].map(title_mapping)
base['Title'] = base['Title'].fillna(0)

## Checking if we missed anything, and general frequencies
print(base['Title'].isna().sum())
print(base['Title'].groupby(base['Title']).count())

## Dropping any significant outliers
for col in base.columns:
    if base[col].isnull().mean()*100>40:
        base.drop(col,axis=1,inplace=True)

## Imputing any remaining values by mean for numbers, and mode for objects
f = lambda x: x.median() if np.issubdtype(x.dtype, np.number) else x.mode().iloc[0]
base = base.fillna(base.groupby('SibSp').transform(f))

## Encoding categorical variables with labels
le=LabelEncoder()
for col in base.columns:
    if base[col].dtypes == object:
        base[col]= le.fit_transform(base[col])

## Validation Data Cleaning
## We apply all of the same steps we used for the testing data

valid.loc[valid.Sex != 'male', 'Sex'] = 1
valid.loc[valid.Sex == 'male', 'Sex'] = 0

valid['Has_Cabin'] = valid["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

valid.Cabin.fillna(0, inplace=True)
valid.loc[valid.Cabin.str[0] == 'A', 'Cabin'] = 4
valid.loc[valid.Cabin.str[0] == 'B', 'Cabin'] = 4
valid.loc[valid.Cabin.str[0] == 'C', 'Cabin'] = 4
valid.loc[valid.Cabin.str[0] == 'D', 'Cabin'] = 3
valid.loc[valid.Cabin.str[0] == 'E', 'Cabin'] = 2
valid.loc[valid.Cabin.str[0] == 'F', 'Cabin'] = 1
valid.loc[valid.Cabin.str[0] == 'G', 'Cabin'] = 1
valid.loc[valid.Cabin.str[0] == 'T', 'Cabin'] = 1
valid['Cabin'] = valid['Cabin'].astype(int)

valid.Fare.fillna(0, inplace=True)
valid.loc[(valid['Fare'] <= 8), 'Fare'] = 0
valid.loc[(valid['Fare'] > 8) & (valid['Fare'] <= 16), 'Fare'] = 1
valid.loc[(valid['Fare'] > 16) & (valid['Fare'] <= 30), 'Fare'] = 2
valid.loc[(valid['Fare'] > 30) & (valid['Fare'] <= 100), 'Fare'] = 3
valid.loc[(valid['Fare'] > 100), 'Fare'] = 4
valid['Fare'] = valid['Fare'].astype(int)


valid['Title'] = valid['Name'].apply(get_title)
    # Group all non-common titles into one single grouping "Rare"
mapping = {'Mlle': 'Rare', 'Major': 'Rare', 'Col': 'Rare', 'Sir': 'Rare', 'Rev': 'Mr',
               'Don': 'Rare', 'Mme': 'Rare', 'Jonkheer': 'Mr', 'Lady': 'Rare',
               'Capt': 'Mr', 'Countess': 'Rare', 'Ms': 'Miss', 'Dona': 'Rare'}
valid.replace({'Title': mapping}, inplace=True)
    # Mapping titles
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
valid['Title'] = valid['Title'].map(title_mapping)
valid['Title'] = valid['Title'].fillna(0)


f = lambda x: x.median() if np.issubdtype(x.dtype, np.number) else x.mode().iloc[0]
valid = valid.fillna(valid.groupby('SibSp').transform(f))

le=LabelEncoder()
for col in valid.columns:
    if valid[col].dtypes == object:
        valid[col]= le.fit_transform(valid[col])


## ---------------- model and data constructors  ---------------- ##


## We conduct a train-test split, separating data we can validate the model against
x_train,x_val = train_test_split(base,test_size=0.20,random_state=42)

## Dataset constructor - Input is a dataframe, output is a dataset object with values encoded into tensors, with the
## required methods for length and retrieving an item by index.
class MyDataset(Dataset):
  
  def __init__(self,df):

    # Separate our explanatory and dependent variables
    x = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Cabin','Has_Cabin','Title']].values
    y = df['Survived'].values

    self.x_train = torch.tensor(x,dtype=torch.float32)
    self.y_train = torch.tensor(y,dtype=torch.float32)
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]

## Neural Net constructor - Input is a batch of tensors with explanatory variables with model specifications, and output are class predictions
## To avoid the vanishing gradient problem, we use rectified linear unit activations on the two hidden layers.
## We pass the linear hidden layers through ReLU activations, and the final linear output layer through a sigmoid activation.
## This in effect is a multi-equation logistic regression for predicting class probabilities. 
class Net(nn.Module):
    def __init__(self, D_in,H1,H2,D_out):
        super(Net,self).__init__()
        self.linear1 = nn.Linear(D_in,H1)
        self.linear2 = nn.Linear(H1,H2)
        self.linear3 = nn.Linear(H2,D_out)
        # self.dropout = nn.Dropout(p=.1)
    
    def forward(self,x):
        x = relu(self.linear1(x))
        # x = self.dropout(x)
        x = relu(self.linear2(x))
        x = sigmoid(self.linear3(x))
        return x

## We call the "Net" class to initialize the model. Net(Input_Dim, Hidden_Layer_1_Neurons, Hidden_Layer_2_Neurons, Output_Dim)
model = Net(10,3,2,1)

## We use binary cross entropy loss for measuring model performance. This is analogous to minimizing MSE in OLS.
## Description:(https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)
criterion = nn.BCELoss()

## After computing the gradients for all tensors in the model, calling optimizer. step() makes the optimizer 
## iterate over all parameters (tensors)it is supposed to update and use their internally stored grad to update their values.
## Learning rate is a key hyperparameter that determines how fast the network moves weights to gradient minima
## Weight decay is an optional hyperparameter which progressivly reduces |weights| each epoch, in effect penalizing overfitting.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

## We call our dataset classes from our train/test split, and returns two dataset objects
train_data = MyDataset(x_train)
val_data = MyDataset(x_val)

## DataLoader takes our dataset objects, and turns them into iterables. 
## Batch-size determines how many row tensors are passed to the model in each epoch.
## Setting shuffle to true, ensures that each batch is a random sample.
train_loader = DataLoader(dataset = train_data, batch_size = 150, shuffle=True)
val_loader = DataLoader(dataset = val_data, batch_size = 100, shuffle = True)


## ---------------- training the model  ---------------- ##

loss_list = []                      ## We initialize two empty lists to append loss from each epoch to
val_loss_list = []
for epoch in range(600):            ## By inputing the range(x), we are choosing 'x' epochs to iterate over the training data
    for x,y in train_loader:        ## Obtain samples for each batch
        optimizer.zero_grad()       ## Zero out the gradient
        y = y.unsqueeze(1)          ## Take targets tensor of shape [150] and coerces it to [150,1] 
        yhat = model(x)             ## Make a prediction
        loss = criterion(yhat,y)    ## Calculate loss
        loss.backward()             ## Differentiate loss w.r.t parameters
        optimizer.step()            ## Update parameters
    
 ## Testing the updated parameters on the held validation data...   
    for w,z in val_loader:          ## Obtain samples for each batch
        z = z.unsqueeze(1)          ## Take targets tensor of shape [150] and coerces it to [150,1]
        y_val_hat = model(w)        ## Make a prediction
        val_loss = criterion(y_val_hat,z)      ## Calculate loss

## At each epoch, we append the calculated loss to a list, so we can graph it's change over time...
    if epoch:
        loss_list.append(loss.item())
        val_loss_list.append(val_loss.item())
print("Finished Training!")

## A simple plotting function for showing loss changes over time as parameters are updated...
plt.plot(loss_list, linewidth=.5)
plt.plot(val_loss_list, linewidth =.5)
plt.legend(("Training Loss", "Validation Loss"))
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.show()


## ---------------- making predictions on the test data and exporting a CSV for submission  ---------------- ##


## Separating the test data into explanatory variables and passengerID indexes
test_data = valid[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Cabin','Has_Cabin','Title']].values
test_passengers = valid['PassengerId']

## Transforming the explanatory variables to a tensor.
x_test_var = Variable(torch.FloatTensor(test_data), requires_grad=True) 

## We initialize a list to place our class predictions in.
survived = []

## We call the model to make the predictions. The parameters from training are saved in "state_dict":
## Which is a dictionary with the optimizer settings, and weights for each parameters. 
# The state_dict is maintained in the background, but can be exported to use in other scripts or in production
test_result = model(x_test_var)

## The test_result predictions from the model are probabilities of class.
## We iterate across the results, and append the predicted class if the probability is greater than .5.
i=0
while i < len(test_result):
    if test_result[i] > .5:
        survived.append(1)
        i += 1
    else:
        survived.append(0)
        i += 1

## We tie together our predictions with passengerID...
submission = [['PassengerId', 'Survived']]
for i in range(len(survived)):
    submission.append([test_passengers[i], survived[i]])

## We write our submission to CSV...
with open('submission.csv', 'w') as submissionFile:
    writer = csv.writer(submissionFile)
    writer.writerows(submission)
    
print('Writing Complete!')

## Et Viola! 
## We have completed creating a neural network from scratch, and applying it to a canonical machine learning dataset. 
## This submission scored .77990 on Kaggle's "Machine Learning from Disaster" competition. (2,627th place of 14,380) 
