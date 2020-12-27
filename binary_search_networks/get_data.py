# Author: Isamu Isozaki
# Date: 2020/11/10
# Purpose: Get data. For details on how the data is preprocessed, please check Analysis.ipynb

from sklearn.model_selection import train_test_split
#imputation. For more details check https://www.kaggle.com/alexisbcook/missing-values
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

def get_data(**args):
    data = pd.read_csv('titanic.csv')
    X, y = data[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']], data['survived']
    X[['age', 'fare']]=X[['age', 'fare']].replace('?', np.nan)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    X[['age', 'fare']] = imp_mean.fit_transform(X[['age', 'fare']])
    X = pd.get_dummies(X, columns=['sex', 'embarked'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args['test_size'], random_state=42)
    return X_train, X_test, y_train, y_test
