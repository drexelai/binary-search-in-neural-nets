#So in this dataset we would be dealing with Churn Modeling i.e. we would be writing a Artificial Neural Network
#to find out reasons as to why and which customers are actually leaving SAP and their dependencies on one another.
#This is a classification problem 0-1 classification(1 if Leaves 0 if customer stays)
#We might use Theano or Tensorflow but the thing is that these libraries require us to write most of the Ml code from
#scratch so instead we make use of Keras which enables us writing powerful Neural Networks with a few lines of code
#Keras runs on Theano and Tensorflow and you can think it of as a Sklearn for Deep Learning
# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# fix random seed for reproducibility
np.random.seed(0)

# Inform the user
print("Looking for dataset...")
# Importing the dataset save it in Pycharm Projects/Name of Project
dataset = pd.read_csv('ChurnModel.csv')
print("Dataset is found...")
print("Dataset is being read.")
print("Preprocessing started...")

#Looking at the features we can see that row no.,name will have no relation with a customer with leaving SAP
#so we drop them from X which contains the features Indexes from 3 to 12
X = dataset.iloc[:, 3:13].values
#We store the Dependent value/predicted value in y by storing the 13th index in the variable y
y = dataset.iloc[:, 13].values
#Printing out the values of X --> Which contains the features
#                           y --> Which contains the target variable

# Encoding categorical data
# Now we encode the string values in the features to numerical values
# The only 2 values are Product Type and Region which need to converted into numerical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()#creating label encoder object no. 1 to encode region name(index 1 in features)
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])#encoding region from string to just 3 no.s 0,1,2 respectively
labelencoder_X_2 = LabelEncoder()#creating label encoder object no. 2 to encode product type name(index 2 in features)
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])#encoding product type from string to just 2 no.s 0,1(onprem,cloud) respectively
#Now creating Dummy variables
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("Preprocessing is over")


# Part 2 - Now let's make the ANN!
'''
Listing out the steps involved in training the ANN with Stochastic Gradient Descent
1)Randomly initialize the weights to small numbers close to 0(But not 0)
2)Input the 1st observation of your dataset in the Input Layer, each Feature in one Input Node
3)Forward-Propagation from Left to Right, the neurons are activated in a way that the impact of each neuron's activation
is limited by the weights.Propagate the activations until getting the predicted result y.
4)Compare the predicted result with the actual result. Measure the generated error.
5)Back-Propagation: From Right to Left, Error is back  propagated.Update the weights according to how much they are
responsible for the error.The Learning Rate tells us by how much such we update the weights.
6)Repeat Steps 1 to 5 and update the weights after each observation(Reinforcement Learning).
Or: Repeat Steps 1 to 5 but update the weights only after a batch of observations(Batch Learning)  
7)When the whole training set is passed through the ANN.That completes an Epoch. Redo more Epochs


'''
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential#For building the Neural Network layer by layer
from keras.layers import Dense#To randomly initialize the weights to small numbers close to 0(But not 0)
from keras.models import model_from_json
# Try to load the model
try:
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    score = model.evaluate(X_train, y_train, verbose=0)
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)#if y_pred is larger than 0.5 it returns true(1) else false(2)
    
except:
    print("Saved model not found\nTraining begins.")
    # Initialising the ANN
    #So there are actually 2 ways of initializing a deep learning model
    #------1)Defining each layer one by one
    #------2)Defining a Graph
    model = Sequential()#We did not put any parameter in the Sequential object as we will be defining the Layers manually

    # Adding the input layer and the first hidden layer
    #This remains an unanswered question till date that how many nodes of the hidden layer do we actually need
    # There is no thumb rule but you can set the number of nodes in Hidden Layers as an Average of the number of Nodes in Input and Output Layer Respectively.
    #Here avg= (11+1)/2==>6 So set Output Dim=6
    #Init will initialize the Hidden Layer weights uniformly
    #Activation Function is Rectifier Activation Function
    #Input dim tells us the number of nodes in the Input Layer.This is done only once and wont be specified in further layers.
    model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

    # Adding the second hidden layer
    model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

    # Adding the output layer
    model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    #Sigmoid activation function is used whenever we need Probabilities of 2 categories or less(Similar to Logistic Regression)
    #Switch to Softmax when the dependent variable has more than 2 categories

    # Compiling the ANN
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fitting the ANN to the Training set
    model.fit(X_train, y_train, batch_size = 10, epochs = 100)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # Save the model by serializing weights to HDF5
    model.save("model.h5")
    print("Training is over.\nSaved model to disk")


    # Part 3 - Making the predictions and evaluating the model
    # Predicting the Test set results
    y_pred = model.predict(X_test)
    print(y_pred)
    y_pred = (y_pred > 0.5)#if y_pred is larger than 0.5 it returns true(1) else false(2)
    # print(y_pred)


print("Start evalution of the model")

# Evaluting the model with different metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve

# print(X_train.shape, X_test.shape, y_pred.shape, y_train.shape, y_test.shape)
acc_score = accuracy_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred) 
area_under_curve = auc(fpr, tpr)
cr = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Evaluation is over.")
print("#"*30)
print("Model Metrics")
print("Accuracy Score: ", acc_score)
print("Area Under Curve: ", area_under_curve)
print("Classification Report:\n", cr)
print("Confusion Matrix:\n", cm)
print("#"*30)


# Part 4: Use the trained model to make predictions for individual test cases
print("Individual test case detected")
print("Load model to test the instance...")
import tensorflow as tf
column_names = ["RowNumber", "CustomerId", "CompanyName", "TQMScore", "Geography", "ProductType", "TotalCustomerYears", "ContractDurationInMonths", "RevenueInMillions", "NumOfProducts", "RenewedBefore", "IsActiveMember", "MaxAttentionContractCostInMillions", "Will Exit?"]
class_names = ["Not exited","Exited"]
test_case = np.array([[10], [15627888], ["Apple"], [580], ["EMEA"], ["Onprem"], [29], [9], [61710.44], [2], [1], [0], [128077.8], [0]])
predict_dataset = np.ndarray(shape=(1,14), buffer=test_case)
X = predict_dataset[:, 2:13]

#We store the Dependent value/predicted value in y by storing the 13th index in the variable y
y = predict_dataset[:, 13]

labelencoder_X_1 = LabelEncoder()#creating label encoder object no. 1 to encode region name(index 1 in features)
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])#encoding region from string to just 3 no.s 0,1,2 respectively
labelencoder_X_2 = LabelEncoder()#creating label encoder object no. 2 to encode product type name(index 2 in features)
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])#encoding product type from string to just 2 no.s 0,1(onprem,cloud) respectively
#Now creating Dummy variables
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# X = X[:, 1:]
predictions = model.predict(X)
print(predictions)
print("#"*30)
print("A single test instance found.\nTest starts...")
for k,v in zip(column_names,test_case):
    print("{}: {}".format(k,v[0]))
print("Probability of leaving SAP: {:.5f}%".format(predictions[0][0]*100))
print("Probability of staying with SAP: {:.5f}%".format(100-predictions[0][0]*100))

# print("Testing company: {}. \n Renewed contract before?: {}\n, Total years as customers: {}\n, Max Attention Contract Cost: {}, \nExit probability: {:.4f}%.".format(test_case[2][0],test_case[2][10],test_case[2][0],test_case[2][0],predictions[0][0]*100))
print("Test finished")
print("#"*30)