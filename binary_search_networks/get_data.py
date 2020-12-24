# Author: Isamu Isozaki
# Date: 2020/11/10
# Purpose: Get data. For details on how the data is preprocessed, please check Analysis.ipynb

from sklearn.model_selection import train_test_split
#imputation. For more details check https://www.kaggle.com/alexisbcook/missing-values
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

def get_titanic_data(**args):
    data = pd.read_csv('titanic.csv')
    X, y = data[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']], data['survived']
    X[['age', 'fare']]=X[['age', 'fare']].replace('?', np.nan)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    X[['age', 'fare']] = imp_mean.fit_transform(X[['age', 'fare']])
    X = pd.get_dummies(X, columns=['sex', 'embarked'])
    print(args)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args['test_size'], random_state=42)
    return X_train, X_test, y_train, y_test


def get_data_churn_rate(**args):
    # data = pd.read_csv('ChurnModel.csv')
    # X, y = data[["TQMScore","Geography","ProductType",	"TotalCustomerYears","ContractDurationInMonths","RevenueInMillions","NumOfProducts"	,"RenewedBefore","IsActiveMember","MaxAttentionContractCost"
    # ]], data['Exited']
    # X = pd.get_dummies(X, columns=['Geography', 'ProductType'])
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args['test_size'], random_state=42)
    dataset = pd.read_csv('ChurnModel.csv')
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
    onehotencoder = OneHotEncoder()
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
    return X_train, X_test, y_train, y_test
