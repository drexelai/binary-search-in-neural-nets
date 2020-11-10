# Author: Isamu Isozaki
# Date: 2020/11/10
# Purpose: Take X, y, and n as input as well as parser arguments. Prints accuracy and return model

from binary_search_networks.model import model

def train(X, y, **args):
    train_model = model(**args)
    train_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # running the fitting
    train_model.fit(X, y, epochs=args['epoch'], batch_size=args['batch_size'], validation_split=args['validation_split'], verbose = args['verbose'])
    _, accuracy = train_model.evaluate(X, y)
    print('Train Accuracy: %.2f' % (accuracy*100))
    return train_model