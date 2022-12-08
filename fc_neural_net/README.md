# Fully Connected Neural Network
## Table of Contents
* [TwoLayerNet](#twolayernet)
  * [`__init__`](#__init__)
  * [`loss`](#loss)
  * [`train`](#train)
  * [`predict`](#predict)
* [NeuralNet](#neuralnet)
  * [`train`](#train)
  * [`init_layers`](#init_layers)
  * [`forward_propagation`](#forward_propagation)
  * [`backward_propagation`](#backward_propagation)
  * [`update`](#update)
  * [Activation functions](#activation-functions)
    * [`sigmoid`](#sigmoid)
    * [`sigmoid_backward`](#sigmoid_backward)
    * [`relu`](#relu)
    * [`relu_backward`](#relu_backward)
  * [Loss functions](#loss-functions)
    * [`get_cost_value`](#get_cost_value)
    * [`convert_prob_into_class`](#convert_prob_into_class)
    * [`get_accuracy_value`](#get_accuracy_value)


[Home](https://github.com/lavinama/conv-first-principles#readme)

## TwoLayerNet

### `__init__`

```python
def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = std * np.random.randn(hidden_size)  + 0.5  # np.zeros(hidden_size) 
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)
```
[Back to top of page](#table-of-contents)
[Home](https://github.com/lavinama/conv-first-principles#readme)

### `loss`

```python
def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
    samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    # Compute the forward pass
    scores = None
    z = np.dot(X, W1) + b1  # (N, num_hidden)
    h = np.maximum(z, 0)    # ReLU
    scores = np.dot(h , W2) + b2 
    # If the targets are not given then jump out, we're done
    if y is None:
        return scores
    # Compute the loss
    loss = 0.0        
    # compute softmax probabilities
    out = np.exp(scores)      # (N, C)
    out /= np.sum(out, axis=1).reshape(N, 1)
    # compute softmax loss
    loss -= np.sum(np.log(out[np.arange(N), y]))
    loss /= N
    loss += 0.5 * reg * (np.sum(W1**2) + np.sum(W2**2))
    # Backward pass: compute gradients
    grads = {}
    # back propagation
    dout = np.copy(out)  # (N, C)
    dout[np.arange(N), y] -= 1
    dh = np.dot(dout, W2.T)
    dz = np.dot(dout, W2.T) * (z > 0)  # (N, H)
    # compute gradient for parameters
    grads['W2'] = np.dot(h.T, dout) / N      # (H, C)
    grads['b2'] = np.sum(dout, axis=0) / N      # (C,)
    grads['W1'] = np.dot(X.T, dz) / N        # (D, H)
    grads['b1'] = np.sum(dz, axis=0) / N       # (H,)
    # add reg term
    grads['W2'] += reg * W2
    grads['W1'] += reg * W1
    return loss, grads
```
[Back to top of page](#table-of-contents) <br />
[Home](https://github.com/lavinama/conv-first-principles#readme)

### `train`

```python
def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
    X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
    after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):                
        random_idxs = np.random.choice(num_train, batch_size)
        X_batch = X[random_idxs]
        y_batch = y[random_idxs]
    
        # Compute loss and gradients using the current minibatch
        loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
        loss_history.append(loss)
        
        self.params['W2'] -= learning_rate * grads['W2']
        self.params['b2'] -= learning_rate * grads['b2']
        self.params['W1'] -= learning_rate * grads['W1']
        self.params['b1'] -= learning_rate * grads['b1']

        if verbose and it % 100 == 0:
            print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

        # Every epoch, check train and val accuracy and decay learning rate.
        if it % iterations_per_epoch == 0:
            # Check accuracy
            train_acc = (self.predict(X_batch) == y_batch).mean()
            val_acc = (self.predict(X_val) == y_val).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            # Decay learning rate
            learning_rate *= learning_rate_decay

    return {
    'loss_history': loss_history,
    'train_acc_history': train_acc_history,
    'val_acc_history': val_acc_history,
    }
```
[Back to top of page](#table-of-contents) <br />
[Home](https://github.com/lavinama/conv-first-principles#readme)

### `predict`

```python
def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
    classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
    the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
    to have class c, where 0 <= c < C.
    """
    y_pred = None
    params = self.params
    z = np.dot(X, params['W1']) + params['b1']
    h = np.maximum(z, 0)
    out = np.dot(h, params['W2']) + params['b2']
    y_pred = np.argmax(out, axis=1)

    return y_pred
```
[Back to top of page](#table-of-contents)
[Home](https://github.com/lavinama/conv-first-principles#readme)

## Neural Net

### `train`
```python
def train(X_train, Y_train, nn_architecture, X_val=None, Y_val=None, num_iters=100, learning_rate=1e-3,
    learning_rate_decay=0.95, batch_size=200, verbose=False, callback=None):
    # initiation of neural net parameters
    params_values = init_layers(nn_architecture, 2)
    num_train = X_train.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)
    # initiation of lists storing the history 
    # of metrics calculated during the learning process 
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    # performing calculations for subsequent iterations
    for it in range(num_iters):
        random_idxs = np.random.choice(num_train, batch_size)
        X_batch = X_train[random_idxs]
        Y_batch = Y_train[random_idxs]
        # step forward
        Y_hat, cache = full_forward_propagation(X_batch, params_values, nn_architecture)
        
        # calculating metrics and saving them in history
        cost = get_cost_value(Y_hat, Y_batch)
        loss_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y_batch)
        train_acc_history.append(accuracy)
        
        # step backward - calculating gradient
        grads_values = full_backward_propagation(Y_hat, Y_batch, cache, params_values, nn_architecture)
        # updating model state
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        
        if verbose and it % 100 == 0:
            print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(it, cost, accuracy))

        # Every epoch, check train and val accuracy and decay learning rate.
        if it % iterations_per_epoch == 0:
            # Check accuracy
            Y_hat_val, cache = full_forward_propagation(X_val, params_values, nn_architecture)
            train_acc = (Y_hat == Y_batch).mean()
            val_acc = (Y_hat_val == Y_val).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            # Decay learning rate
            learning_rate *= learning_rate_decay
            
    return params_values
```
[Back to top of page](#table-of-contents)
[Home](https://github.com/lavinama/conv-first-principles#readme)

### `init_layers`
```python
def init_layers():
    nn_architecture = [
    {"input_dim": 2, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"}]
    # random seed initiation
    np.random.seed(1)
    # number of layers in our neural network
    number_of_layers = len(nn_architecture)
    # parameters storage initiation
    params_values = {}
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        
        # extracting the number of units in layers
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        # initiating the values of the W matrix
        # and vector b for subsequent layers
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
    return params_values
```
[Back to top of page](#table-of-contents)
[Home](https://github.com/lavinama/conv-first-principles#readme)

### `forward_propagation`
```python
def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    # calculation of the input value for the activation function
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    # selection of activation function
    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')
        
    # return of calculated activation A and the intermediate Z matrix
    return activation_func(Z_curr), Z_curr

def full_forward_propagation(X, params_values, nn_architecture):
    # creating a temporary memory to store the information needed for a backward step
    memory = {}
    # X vector is the activation for layer 0 
    A_curr = X
    
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        # transfer the activation from the previous iteration
        A_prev = A_curr
        
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        # extraction of W for the current layer
        W_curr = params_values["W" + str(layer_idx)]
        # extraction of b for the current layer
        b_curr = params_values["b" + str(layer_idx)]
        # calculation of activation for the current layer
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        
        # saving calculated values in the memory
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
       
    # return of prediction vector and a dictionary containing intermediate values
    return A_curr, memory
```
[Back to top of page](#table-of-contents)
[Home](https://github.com/lavinama/conv-first-principles#readme)

### `backward_propagation`
```python
def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    # number of examples
    m = A_prev.shape[1]
    
    # selection of activation function
    if activation is "relu":
        backward_activation_func = relu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')
    
    # calculation of the activation function derivative
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    
    # derivative of the matrix W
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    # derivative of the vector b
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    # derivative of the matrix A_prev
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    
    # number of examples
    m = Y.shape[1]
    # a hack ensuring the same shape of the prediction vector and labels vector
    Y = Y.reshape(Y_hat.shape)
    
    # initiation of gradient descent algorithm
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        # we number network layers from 1
        layer_idx_curr = layer_idx_prev + 1
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    return grads_values
```
[Back to top of page](#table-of-contents)
[Home](https://github.com/lavinama/conv-first-principles#readme)

### `update`
```python
def update(params_values, grads_values, nn_architecture, learning_rate):
    # iteration over network layers
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]
    return params_values
```
[Back to top of page](#table-of-contents)
[Home](https://github.com/lavinama/conv-first-principles#readme)

### Activation functions

#### `sigmoid`
```python
def sigmoid(Z):
    return 1/(1+np.exp(-Z))
```
[Back to top of page](#table-of-contents)
[Home](https://github.com/lavinama/conv-first-principles#readme)

#### `sigmoid_backward`
```python
def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)
```
[Back to top of page](#table-of-contents)
[Home](https://github.com/lavinama/conv-first-principles#readme)

#### `relu`
```python
def relu(Z):
    return np.maximum(0,Z)
```
[Back to top of page](#table-of-contents)
[Home](https://github.com/lavinama/conv-first-principles#readme)

#### `relu_backward`
```python
def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ
```
[Back to top of page](#table-of-contents)
[Home](https://github.com/lavinama/conv-first-principles#readme)

### Loss functions

#### `get_cost_value`
```python
def get_cost_value(Y_hat, Y):
    # number of examples
    m = Y_hat.shape[1]
    # calculation of the cost according to the formula
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)
```
[Back to top of page](#table-of-contents)
[Home](https://github.com/lavinama/conv-first-principles#readme)

#### `convert_prob_into_class`
```python
# an auxiliary function that converts probability into class
def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_
```
[Back to top of page](#table-of-contents)
[Home](https://github.com/lavinama/conv-first-principles#readme)

#### `get_accuracy_value`
```python
def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()
```
[Back to top of page](#table-of-contents)
[Home](https://github.com/lavinama/conv-first-principles#readme)