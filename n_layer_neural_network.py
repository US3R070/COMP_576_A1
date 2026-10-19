__author__ = 'Allen_Wu'
import argparse
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

from three_layer_neural_network_2026 import NeuralNetwork, plot_decision_boundary

def generate_data(dataset='moons'):
    '''
    generate data
    :param dataset: 'moons' or 'circles'
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    if dataset == 'moons':
        X, y = datasets.make_moons(200, noise=0.20)
    elif dataset == 'circles':
        X, y = datasets.make_circles(200, noise=0.2, factor=0.6)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return X, y

class Layer(object):
    """
    This class builds a single layer of a neural network (Hint 3)
    """
    def __init__(self, input_dim, output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param input_dim: input dimension
        :param output_dim: output dimension
        :param actFun_type: type of activation function. Options: 'tanh', 'sigmoid', 'relu', or None
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        
        np.random.seed(seed)
        self.W = np.random.randn(self.input_dim, self.output_dim) / np.sqrt(self.input_dim)
        self.b = np.zeros((1, self.output_dim))
        
        self.X = None
        self.z = None

    def actFun(self, z, type):
        if type == 'tanh':
            return np.tanh(z)
        elif type == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-z))
        elif type == 'relu':
            return np.maximum(0, z)
        return z

    def diff_actFun(self, z, type):
        if type == 'tanh':
            return 1.0 - np.tanh(z) ** 2
        elif type == 'sigmoid':
            sig = 1.0 / (1.0 + np.exp(-z))
            return sig * (1.0 - sig)
        elif type == 'relu':
            return (z > 0).astype(float)
        return np.ones_like(z)

    def feedforward(self, X):
        '''
        Feedforward for a single layer (Hint 3 & 4)
        :param X: input data to this layer
        :return: activation output of this layer
        '''
        self.X = X
        self.z = np.dot(X, self.W) + self.b
    
        if self.actFun_type is not None:
            return self.actFun(self.z, self.actFun_type)
        return self.z

    def backprop(self, d_next):
        '''
        Backprop for a single layer (Hint 3 & 5)
        :param d_next: upstream gradient (dL / da if activation exists, or dL / dz)
        :return: dW, db, dX (gradient wrt input X to pass to previous layer)
        '''
        if self.actFun_type is not None:
            dz = d_next * self.diff_actFun(self.z, self.actFun_type)
        else:
            dz = d_next
            
        dW = np.dot(self.X.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        dX = np.dot(dz, self.W.T)
        
        return dW, db, dX


class DeepNeuralNetwork(NeuralNetwork):
    """
    This class builds a deep neural network inheriting NeuralNetwork (Hint 1)
    """
    def __init__(self, num_layers, layer_size, input_dim=2, output_dim=2, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param num_layers: total number of layers (including hidden layers and output layer)
                           e.g., num_layers=3 means 2 hidden layers + 1 output layer (or total weight layers)
        :param layer_size: size of the hidden layers (can be an int, or a list of ints)
        :param input_dim: dimension of input features
        :param output_dim: dimension of output classes
        :param actFun_type: 'tanh', 'sigmoid', or 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''

        super(DeepNeuralNetwork, self).__init__(
            nn_input_dim=input_dim,
            nn_hidden_dim=(layer_size[0] if isinstance(layer_size, list) else layer_size),
            nn_output_dim=output_dim,
            actFun_type=actFun_type,
            reg_lambda=reg_lambda,
            seed=seed
        )
        
        self.num_layers = num_layers
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        
        num_hidden = max(1, num_layers - 1)
        if isinstance(layer_size, int):
            hidden_dims = [layer_size] * num_hidden
        elif isinstance(layer_size, list):
            if len(layer_size) != num_hidden:
                
                hidden_dims = [layer_size[i % len(layer_size)] for i in range(num_hidden)]
            else:
                hidden_dims = layer_size
        else:
            raise ValueError("layer_size must be either an int or a list of ints")
            
        self.layer_dims = [input_dim] + hidden_dims + [output_dim]
        
        
        self.layers = []
        for i in range(len(self.layer_dims) - 1):
            # 隱藏層使用指定 activation，輸出層不使用（留給 softmax）
            cur_actFun_type = self.actFun_type if i < len(self.layer_dims) - 2 else None
            layer = Layer(
                input_dim=self.layer_dims[i],
                output_dim=self.layer_dims[i + 1],
                actFun_type=cur_actFun_type,
                reg_lambda=self.reg_lambda,
                seed=seed + i
            )
            self.layers.append(layer)

    def feedforward(self, X):

        A = X
        for layer in self.layers:
            A = layer.feedforward(A)
            
        exp_scores = np.exp(A - np.max(A, axis=1, keepdims=True))  # 數值穩定性防 overflow
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backprop(self, X, y):

        num_examples = len(X)
        
        # Softmax + Cross-entropy 的導數: p - y
        dz = self.probs.copy()
        dz[range(num_examples), y] -= 1
        
        grads = []
        # 從後往前依序反向傳播
        for layer in reversed(self.layers):
            dW, db, dz = layer.backprop(dz)
            grads.insert(0, (dW, db))
            
        return grads

    def calculate_loss(self, X, y):

        num_examples = len(X)
        self.feedforward(X)
        
        correct_labels = self.probs[range(num_examples), y]
        data_loss = -np.sum(np.log(np.maximum(correct_labels, 1e-15)))
        
        # L2 正則化損失
        reg_loss = sum(np.sum(np.square(layer.W)) for layer in self.layers)
        data_loss += self.reg_lambda / 2.0 * reg_loss
        
        return (1.0 / num_examples) * data_loss

    def predict(self, X):
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        Fit model using gradient descent, including L2 regularization derivative (Hint 2 & 6)
        '''
        for i in range(0, num_passes):
            self.feedforward(X)
            grads = self.backprop(X, y)

            for layer, (dW, db) in zip(self.layers, grads):
                # 加上 L2 weight regularization 的導數
                dW_reg = dW + self.reg_lambda * layer.W
                layer.W -= epsilon * dW_reg
                layer.b -= epsilon * db
                
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        plot_decision_boundary(lambda x: self.predict(x), X, y)


def main():
    parser = argparse.ArgumentParser(description="Train an n-layer deep neural network.")
    parser.add_argument('--num_layers', type=int, default=3, help='Total number of layers (e.g. 3 = 2 hidden + 1 output)')
    parser.add_argument('--layer_size', type=int, default=5, help='Number of hidden units per hidden layer')
    parser.add_argument('--actFun_type', type=str, default='tanh', choices=['tanh', 'sigmoid', 'relu'], help='Activation function')
    parser.add_argument('--dataset', type=str, default='moons', choices=['moons', 'circles'], help='Dataset: moons or circles')
    parser.add_argument('--epochs', type=int, default=20000, help='Number of training iterations')
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    print(f"=== Training Configuration ===")
    print(f"Dataset: {args.dataset}")
    print(f"Number of layers: {args.num_layers}")
    print(f"Layer size (hidden dim): {args.layer_size}")
    print(f"Activation: {args.actFun_type}")
    print(f"Iterations: {args.epochs}, Learning rate: {args.lr}")
    print("==============================\n")

    X, y = generate_data(dataset=args.dataset)

    model = DeepNeuralNetwork(
        num_layers=args.num_layers,
        layer_size=args.layer_size,
        input_dim=X.shape[1],
        output_dim=len(np.unique(y)),
        actFun_type=args.actFun_type,
        reg_lambda=0.01
    )

    model.fit_model(X, y, epsilon=args.lr, num_passes=args.epochs, print_loss=True)
    model.visualize_decision_boundary(X, y)

if __name__ == "__main__":
    main()