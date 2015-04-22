import theano, theano.tensor as T
import numpy as np

def quadratic_form(tensor, x, y):
    return (T.dot(tensor, x.T) * y.T).sum(axis=1).T

def facebook_quadratic_form(U, V, x, y):
    return U.dot(x).dot((V.dot(y).T)).diagonal()

def create_shared(name="", *dims):
    return theano.shared(np.random.uniform(
            low=-1.0 / np.sqrt(dims[0]),
            high=1.0 / np.sqrt(dims[0]), size=dims).astype(theano.config.floatX), name=name)
