import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        # Set activation function f(x) = sigmoid
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))
                    
    def train(self, features, targets):

        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            # Implement the forward pass function below
            final_outputs, hidden_outputs = self.forward_pass_train(X)  
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        
        # This is output of input layer
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        # Activation
        hidden_outputs = self.activation_function(hidden_inputs)

        # This is output of hidden layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        # Project guide required final activation f(x) = x
        final_outputs = final_inputs
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):

        # error_term = (y-y_hat) * f(h)_prime
        # For sigmoid, f(h)_prime = f(h) * (1 - f(h))
        # For final term, activation function f(x) = x, f(x)_prime = 1
        
        error = y- final_outputs  
        output_error_term = error
        # output_error_term = error * final_outputs * (1 - final_outputs)
        
        hidden_error = np.dot(output_error_term, self.weights_hidden_to_output.T)
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        
        # Weight step (input to hidden)
        delta_weights_i_h += self.lr * hidden_error_term * X[:, None]
        # Weight step (hidden to output)
        delta_weights_h_o += self.lr * output_error_term * hidden_outputs[:, None]
        
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):

        # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += delta_weights_h_o / n_records 
        # update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += delta_weights_i_h / n_records 

    def run(self, features):

        # This calculation step that apply trained weight & bias
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs
        
        return final_outputs


#########################################################
# Set hyperparameters here
##########################################################
iterations = 6500
learning_rate = 0.5
hidden_nodes = 25
output_nodes = 1
