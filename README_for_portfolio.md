# DEEP LEARNING NANODEGREE project1.Predicting_Bike_Sharing_patterns

# 1. Abstract

### 1) Purpose

The purpose of this project is to predict number of bike rental for days, hours

### 2) Input data

Input data is csv file that includes number of bike sharing for days, hours, weekdays, weather etc

For feeding this data to Neuaral Net and tunning weights, we need to some data tunnings


# 2. Background Learning

### 1) Introduction to Neural Net

① asdaklsfolqnas
  
② asdkljqf

### 2) Gradient Descent

① asdaklsfolqnas
  
② asdkljqf

### 3) Training Neural Network

① asdaklsfolqnas
  
② asdkljqf


# 3. Code Flow

Firstly, we need to preprocess data

We don't have to use all of data we have

So we have to check what important variables is, and devide into three parts (train, validation, test)

Secondly, we will use preprocessed data to predict result

We have to decide the architecture of NeuralNet, size, loss function, ...

I had to spend much time to tuning hyperparameters like learning rate, number of epochs, size of hidden nodes


### 1) Preparing Data

① Loading and preparing data

```python
data_path = 'Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)

rides.head()
```

<img src="./images/input_data_1.png" width="800">

② Checking out the data

I checked and plotted for 10 days data

```python
rides[:24*10].plot(x='dteday', y='cnt')
```

<img src="./images/input_data_2.png" width="400">

③ Dummify variables

For example, month has 12 values (1~12)

But December doesn't mean that it has much valuable than January

Because we put numbers to our Neural Net by x (input layer),
we need to set all variables to equal value (0 or 1)

And that variables are season, weather, month, hour, weekday

```python
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
data.head()
```

<img src="./images/dummify_variables.png" width="800">

④ Scaling target variables

For example, number of rental for an hour can be 0 or even 10,000

It can have too much difference and it can make distortion for Neuaral Net calculation

So we have to scaling to equal range that have 0 mean and 1 standard deviation

That variables are total rental number, registered number, casueal number, temperature, humidity, windspeed

```python
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std
```

<img src="./images/scaling_target_variables.png" width="800">

⑤ Splitting data into training, validation, testing sets

```python
test_data = data[-21*24:]

data = data[:-21*24]

target_fields = ['cnt', 'casual', 'registered']

features, targets = data.drop(target_fields, axis=1), data[target_fields]

test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

train_features, train_targets = features[:-60*24], targets[:-60*24]

val_features, val_targets = features[-60*24:], targets[-60*24:]
```

<img src="./images/test_features.png" width="800">
<img src="./images/test_targets.png" width="200">


### 2) Training , valiation, test

I will train this data to my NeuralNet architecture and simultaneously check loss for train & validation data

After complete tuning all parameters, I will check arruracy for test data only once

① Train data

```python

from my_answers import NeuralNetwork
import sys
from my_answers import iterations, learning_rate, hidden_nodes, output_nodes

N_i = train_features.shape[1]

###### I will provide this code explanation blow ######
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for ii in range(iterations):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']
    
    ###### I will provide this code explanation blow ######
    network.train(X, y)
    
    # Printing out the training progress
    train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)
    
```

② Explanation for NeuralNetwork initiation

```python

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
        
        # I selected activation function f(x) = sigmoid
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))
        
```

③ Explanation for NeuralNetwork.train method

```python

class NeuralNetwork(object):
    def train(self, features, targets):

        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            # Implement the forward pass
            final_outputs, hidden_outputs = self.forward_pass_train(X)  
            # Implement the backproagation
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
            
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)
```
```python

    def forward_pass_train(self, X):

        # signals into hidden layer
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) 
        # signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) 

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        # Project guide required final activation f(x) = x
        # signals from final output layer
        final_outputs = final_inputs 
        
        return final_outputs, hidden_outputs
```
```python
    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):

        # Output layer error is the difference between desired target and actual output.
        error = y- final_outputs       
        output_error_term = error # * final_outputs * (1 - final_outputs)
        
        hidden_error = np.dot(output_error_term, self.weights_hidden_to_output.T)
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        
        # Weight step (input to hidden)
        ### changed after first feedback (190523) ###
        ### delta_weights_i_h += self.lr * hidden_error_term * X[:, None]
        delta_weights_i_h += hidden_error_term * X[:, None]
        
        # Weight step (hidden to output)
        ### changed after first feedback (190523) ###
        ### delta_weights_h_o += self.lr * output_error_term * hidden_outputs[:, None]
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]
        
        return delta_weights_i_h, delta_weights_h_o
```

# 4. Result

| Trial |   Learning Rate   |   Hidden Nodes   |   Iteration   |              Explanation                  |
| ----- | -------           | -------          |  -------      |                 ------                    |
|  1    |        0.1        |        2         |       100     | Loss gets down, it means bad architecture |
|  2    |        0.1        |        100       |       100     | Validation loss exploded, need lower hidden nodes |
|  3    |        0.1        |        50        |       100     | Validation loss increased slowly, need lower hidden nodes |
|  4    |        0.1        |        25        |       100     | Validation loss decreased slowly, need lower hidden nodes |
|  5    |        0.1        |        13        |       100     | Validation loss increased slowly, need higher hidden nodes |
|  6    |        0.1        |        20        |       100     | Validation loss increased slowly, need higher hidden nodes |
|  7    |        0.1        |        30        |       100     | Validation loss increased slowly, Hidden nodes should be 25 |
|  8    |        0.5        |        25        |       100     | Increased learning rate was effective, need to be higher one |
|  9    |        1.0        |        25        |       100     | Loss exploded, learning rate should be 0.5 |
|  10   |        0.5        |        25        |       7000    | Seeing loss graph, val loss is inclined after 6500 iterations|
|  11   |        0.5        |        25        |       6500    | This is optimum hyperparameter I tuned |

*Graph*

<img src="./images/loss/loss_train_val_11.png" width="400">
<img src="./images/result/result_11_lr=0.5,hidden=25,iteration=6500.png" width="800">



# 5. Conclusion & Discussion

### 1) Meaning

I already used Keras and TensorFlow library at Self Driving Car Nanodegree program

Especially at Keras, it was possible to implement not only simple Neural Net but also complex and famous architectures

even if I don't understand principle of Neural Net

At that time, I thought I understood everything of Neural Net

But this time was very good chance for me to studying Neural Net

especially the mathmatical principle of Forward propagation, Backpropagation

### 2) About architecture

This project confined architectures to just 1 hidden layer Neural Net

So there were only 3 variables I can adjust

It was learning rate, number of hidden nodes, iterations

Of course on the contrary, thanks to it, it was good time to feel the effects of that variables

But if I can change architecture more complex, I will get better result



