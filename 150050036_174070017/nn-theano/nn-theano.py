

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import matplotlib
import theano
import graphviz
import theano.tensor as T
import pydot
from IPython.display import Image
from IPython.display import SVG
import timeit
import os
print(os.path.expanduser('~/.theanorc.txt'))
print(theano.config.device)
# theano.config.device = 'opencl0:0'
# theano.config.floatX = 'float32'
# theano.config.cnmem = 100

np.random.seed(0)
train_X, train_y = sklearn.datasets.make_moons(200, noise=0.20)
train_X = train_X.astype(np.float32)
train_y = train_y.astype(np.int32)
train_y_onehot = np.eye(2)[train_y]
plt.scatter(train_X[:,0], train_X[:,1], s=40, c=train_y, cmap=plt.cm.Spectral)


def plot_decision_boundary(pred_func):
    x_min, x_max = train_X[:, 0].min() - .5, train_X[:, 0].max() + .5
    y_min, y_max = train_X[:, 1].min() - .5, train_X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap=plt.cm.Spectral)



num_examples = len(train_X) 
nn_input_dim = 2 
nn_output_dim = 2
nn_hdim = 10 

epsilon = np.float32(0.01) 
reg_lambda = np.float32(0.01) 

X = theano.shared(train_X.astype('float32'))
y = theano.shared(train_y_onehot.astype('float32'))

W1 = theano.shared(np.random.randn(nn_input_dim, nn_hdim).astype('float32'), name='W1')
b1 = theano.shared(np.zeros(nn_hdim).astype('float32'), name='b1')
W2 = theano.shared(np.random.randn(nn_hdim, nn_output_dim).astype('float32'), name='W2')
b2 = theano.shared(np.zeros(nn_output_dim).astype('float32'), name='b2')

z1 = X.dot(W1) + b1
a1 = T.tanh(z1)
z2 = a1.dot(W2) + b2
y_hat = T.nnet.softmax(z2)

loss_reg = 1./num_examples * reg_lambda/2 * (T.sum(T.sqr(W1)) + T.sum(T.sqr(W2))) 
loss = T.nnet.categorical_crossentropy(y_hat, y).mean() + loss_reg
prediction = T.argmax(y_hat, axis=1)

# Gradients
dW2 = T.grad(loss, W2)
db2 = T.grad(loss, b2)
dW1 = T.grad(loss, W1)
db1 = T.grad(loss, b1)

forward_prop = theano.function([], y_hat)
calculate_loss = theano.function([], loss)
predict = theano.function([], prediction)

gradient_step = theano.function(
    [],
    updates=((W2, W2 - epsilon * dW2),
             (W1, W1 - epsilon * dW1),
             (b2, b2 - epsilon * db2),
             (b1, b1 - epsilon * db1)))

def build_model(num_passes=1000, print_loss=False):
    np.random.seed(0)
    W1.set_value((np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)).astype('float32'))
    b1.set_value(np.zeros(nn_hdim).astype('float32'))
    W2.set_value((np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)).astype('float32'))
    b2.set_value(np.zeros(nn_output_dim).astype('float32'))
    
    for i in range(0, num_passes):
        gradient_step()
        
        if print_loss:
            print("Loss after iteration %i: %f" %(i, calculate_loss()))

# Build a model with a 3-dimensional hidden layer
build_model(print_loss=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(x))
plt.title("Decision Boundary for hidden layer size 3")

