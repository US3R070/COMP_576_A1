__author__ = 'Allen_Wu'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

class Layer(object):
    """
    This class builds a layer of neural network
    """
    def __init__(self, input_dim, output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param input_dim: input dimension
        :param output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        
        self.W = np.random.randn(self.input_dim, self.output_dim) / np.sqrt(self.input_dim)
        self.b = np.zeros((1, self.output_dim))
        
        self.X = None
        self.z = None

    def feedforward(self, X):
        '''
        feedforward
        :param X: input data
        :return: output of the layer
        '''
        self.X = X
        self.z = np.dot(X, self.W) + self.b
    
        if self.actFun_type is not None:
            return self.actFun(self.z, self.actFun_type)
        
        return self.z
    
    def actFun(self, z, type):

        if type == 'tanh':
            return np.tanh(z)
        elif type == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif type == 'relu':
            return np.maximum(0, z)
        
        return None

    def diff_actFun(self, z, type):
        
        if type == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif type == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
        elif type == 'relu':
            return (z > 0).astype(float)
        
        return None
    
    def calculate_loss(self, X, y):
        '''
        calculate_loss compute the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X, self.actFun)
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
        correct_labels = self.probs[range(num_examples), y]
        data_loss = -np.sum(np.log(correct_labels))

        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        
        return (1. / num_examples) * data_loss
    
    
    def backprop(self, d_next):
        '''
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        '''
        
        # IMPLEMENT YOUR BACKPROP HERE
        
        
        if self.actFun_type is not None:
            d_next = d_next * self.diff_actFun(self.z, self.actFun_type)
            
        dW = np.dot(self.X.T, d_next) + self.reg_lambda * self.W
        db = np.sum(d_next, axis=0, keepdims=True)
        
        dX = np.dot(d_next, self.W.T)
        
        return dW, db, dX


class DeepNeuralNetwork(object):
    """
    This class builds a deep neural network
    """
    def __init__(self, layer_dims, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param layer_dims: list of layer dimensions
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.layer_dims = layer_dims
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        
        self.layers = []
        
        # add layers to the network
        for i in range(len(layer_dims) - 1):
            cur_actFun_type = self.actFun_type if i < len(layer_dims) - 2 else None
            layer = Layer(input_dim=layer_dims[i], output_dim=layer_dims[i + 1], actFun_type=cur_actFun_type, reg_lambda=self.reg_lambda, seed=seed)
            self.layers.append(layer)
            
    def feedforward(self, X):
        
        A = X

        for layer in self.layers:
            A = layer.feedforward(A)
            
        exp_scores = np.exp(A)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs
    
    def backprop(self, X, y):
        num_examples = len(X)
        
        dz = self.probs.copy()
        dz[range(num_examples), y] -= 1
        
        grads = []
        
        for layer in reversed(self.layers):
            dW, db, dz = layer.backprop(dz)  
            grads.insert(0, (dW, db))
            
        return grads
            
    def calculate_loss(self, X, y):
        num_examples = len(X)
        self.feedforward(X)
        
        correct_labels = self.probs[range(num_examples), y]
        data_loss = -np.sum(np.log(correct_labels))
        reg_loss = sum(np.sum(np.square(layer.W)) for layer in self.layers)
        data_loss += self.reg_lambda / 2 * reg_loss
        
        return (1. / num_examples) * data_loss
    
    def predict(self, X):
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)
    
    def fit_model(self, X, y, epsilon=0.01, num_passes=30000, print_loss=True):
        for i in range(0, num_passes):
            self.feedforward(X)
            grads = self.backprop(X, y)

            for layer, (dW, db) in zip(self.layers, grads):
                layer.W -= epsilon * dW
                layer.b -= epsilon * db
                
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))        

    def visualize_decision_boundary(self, X, y):
            plot_decision_boundary(lambda x: self.predict(x), X, y)
   
def runs(model, X, y, epsilon=0.01, num_passes=24000, print_loss=True):
    model.fit_model(X, y, epsilon=epsilon, num_passes=num_passes, print_loss=print_loss)
    model.visualize_decision_boundary(X, y)
    
def main():
    
    
    
    X, y = generate_data()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()
    
    # runs(DeepNeuralNetwork(layer_dims=[2, 10, 2], actFun_type='tanh', reg_lambda=0.01), X, y)
    # runs(DeepNeuralNetwork(layer_dims=[2, 10, 5, 2], actFun_type='tanh', reg_lambda=0.01), X, y)
    runs(DeepNeuralNetwork(layer_dims=[2,80,40,20,10,5, 2], actFun_type='tanh', reg_lambda=0.01), X, y)

if __name__ == "__main__":
    
    main()