# Author: Isamu Isozaki
# Date: 2020/11/10
# Purpose: Take X, y, and n as input as well as parser arguments. Prints accuracy and return model and train accuracy

from binary_search_networks.model import model
from keras.callbacks import EarlyStopping

def train(X, y, **args):
    train_model = model(**args)
    train_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # running the fitting
    callback = EarlyStopping(monitor='loss', patience=args['early_stopping_patience'])
    if args['no_early_stopping']:
        history = train_model.fit(X, y, epochs=args['epoch'], batch_size=args['batch_size'], validation_split=args['validation_split'], verbose = args['verbose'])
    else:
        history = train_model.fit(X, y, epochs=args['epoch'], batch_size=args['batch_size'], validation_split=args['validation_split'], verbose = args['verbose'], callbacks=[callback])
    val_accuracy = history.history["val_accuracy"][-1]
    _, accuracy = train_model.evaluate(X, y)
    return train_model, accuracy, val_accuracy