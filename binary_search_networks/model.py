# Author: Isamu Isozaki
# Date: 2020/11/10
# Purpose: construct model given input n

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout

# Setup model given hidden layer with dimension input dimension*n
def model(**args):
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Dense(args['n'], input_dim=11, activation='relu'))
    model.add(Dropout(args['dropout_rate']))
    model.add(Dense(1, activation='sigmoid'))
    return model