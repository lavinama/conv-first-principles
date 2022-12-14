import numpy as np
from conv_1d.conv_1d import *

def initialize_parameters():
    np.random.seed(1)                             # so that your "random" numbers match ours        
    W = np.random.randn(3) # (k)
    b = np.random.randn(1) # (1)
    parameters = {"W1": W, "b1": b}
    return parameters

def forward_propagation(X, parameters):
    """
    CONV1D -> RELU    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters
    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    # creating a temporary memory to store the information needed for a backward step
    memory = {}
    # X vector is the activation for layer 0 
    A_curr = X
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters["W1"]
    # CONV1D: stride of 1, padding of 1
    Z1, cache = conv_1D_forward(A_curr, W1)
    # saving calculated values in the memory
    memory["cache1"] = cache

    # RELU
    A1 = Z1 * (Z1 > 0)
    memory["A1"] = A1
    
    return A1, memory

def backward_propagation(Y_hat, Y, memory):
    grads_values = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)

    dA_prev = MSE_loss_grad(Y_hat, Y)
    dA_curr = dA_prev

    # calculation of the activation function derivative
    dZ_curr = relu_backward(dA_curr, Y_hat)
    cache1 = memory["cache1"]
    dA_prev, dW_curr, db_curr = conv_1D_backward(dZ_curr, cache1)
    grads_values["dW1"] = dW_curr
    grads_values["db1"] = db_curr

    return grads_values

def update(params_values, grads_values, learning_rate):
    # iteration over network layers
    params_values["W1"] -= learning_rate * grads_values["dW1"]        
    params_values["b1"] -= learning_rate * grads_values["db1"]
    return params_values

def MSE_loss(predictions, targets):
    """
    Computes Mean Squared error/loss between targets
    and predictions. 
    Input: predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
           targets (N, k) ndarray     (N: no. of samples, k: no. of output nodes)
    Returns: scalar
    Note: The averaging is only done over the output nodes and not over the samples in a batch.
    Therefore, to get an answer similar to PyTorch, one must divide the result by the batch size.
    """
    return np.sum((predictions-targets)**2)/predictions.shape[1]

def MSE_loss_grad(predictions, targets):
    """
    Computes mean squared error gradient between targets 
    and predictions. 
    Input: predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
           targets (N, k) ndarray     (N: no. of samples, k: no. of output nodes)
    Returns: (N,k) ndarray
    Note: The averaging is only done over the output nodes and not over the samples in a batch.
    Therefore, to get an answer similar to PyTorch, one must divide the result by the batch size.
    """
    return 2*(predictions-targets)/predictions.shape[1]

def relu(Z):
    return np.maximum(0,Z)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def train(X_train, Y_train, X_val=None, Y_val=None, learning_rate = 0.001, learning_rate_decay=0.95,
     num_iters=20, batch_size=4, verbose=False):
    """
    CONV1D -> RELU -> MAXPOOL
    
    Arguments:
    X_train -- training set, of shape (None, 64)
    Y_train -- test set, of shape (None, n_y = 1)
    X_test -- training set, of shape (None, 64)
    Y_test -- test set, of shape (None, n_y = 1)    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    np.random.seed(1)                            # to keep results consistent (numpy seed)                       
    num_train = X_train.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)
    train_loss_history = []                                        # To keep track of the cost
    val_loss_history = []

    # Create Placeholders of the correct shape
    # X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    # performing calculations for subsequent iterations
    for it in range(num_iters):
        random_idxs = np.random.choice(num_train, batch_size)
        X_batch = X_train[random_idxs]
        Y_batch = Y_train[random_idxs]
        # Forward propagation: Build the forward propagation in the tensorflow graph
        Y_hat, memory = forward_propagation(X_batch, parameters)
        
        # calculating metrics and saving them in history
        loss = MSE_loss(Y_hat, Y_batch)
        train_loss_history.append(loss)
        
        # Backpropagation
        grads_values = backward_propagation(Y_hat, Y_batch, memory)
        
        # updating model state
        parameters = update(parameters, grads_values, learning_rate)

        if verbose and it % 5 == 0:
            print('iteration %d / %d: loss %f' % (it, num_iters, loss))
        if X_val and it % iterations_per_epoch == 0:
            Y_val_hat, _ = forward_propagation(X_test, parameters)
            val_loss_history.append(MSE_loss(Y_val_hat, Y_val))
    
    losses = {"train_loss": train_loss_history,
            "val_loss": val_loss_history}
    
    return parameters, losses
    
if __name__ == "__main__":
    X = np.random.randn(20, 10) # (N, n_W)
    # Generate noise samples
    noise = np.random.normal(0, np.sqrt(10), size=[20,8])
    Y = [np.square(X[:,0:8])*np.cos(X[:,0:8]) + 4*X[:,0:8]] + noise
    Y = Y.reshape(Y.shape[1], Y.shape[2])
    X_train = X[:16, :]
    X_test = X[16:, :]
    Y_train = Y[:16, :]
    Y_test = Y[16:, :]
    # Training
    parameters, losses = train(X_train, Y_train)
    # Prediction
    Y_test_hat, _ = forward_propagation(X_test, parameters)
    print("loss:", MSE_loss(Y_test_hat, Y_test))