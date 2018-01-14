import numpy as np

np.seterr(all='log')

def relu_activation(Z):
    cache = Z
    A = np.maximum(0, Z)
    assert  A.shape == Z.shape
    
    return A, cache

def sigmoid_activation(Z):
    cache = Z
    A = 1 / (1 + np.exp(-Z))
    assert A.shape == Z.shape

    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    assert dZ.shape == Z.shape

    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert dZ.shape == Z.shape

    return dZ

def encode_one_hot(targets, num_classes):
    """
        Function used to convert a numpy array with labels to a feature/class vector
        Arguments:
        targets - the array that needs to be encoded
        num_classes - number for rows/classes in the resulting vector
    """
    return np.eye(num_classes)[np.array(targets)]

def get_data():
    """
        Loads the CIFAR-10 dataset
        Creates corresponding feature/class vectors for the labels
        Returns the data and labels for both training and set sets
    """
    import pickle
    X_train_list = []
    labels_list = []
    X_test = None
    Y_test = None

    for i in range(1, 6):
        with open('datasets/data_batch_' + str(i), 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
            X_train_list.append(d[b'data'])
            labels_list.append(d[b'labels'])

    with open('datasets/test_batch', 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
        X_test = np.array(d[b'data'])
        test_labels = d[b'labels']
        Y_test = encode_one_hot(test_labels, (np.max(test_labels) + 1))


    X_train = np.concatenate(X_train_list)
    X_train = X_train / 255

    #One hot encoding to get correct labels matrix
    labels_list = np.concatenate(labels_list)
    Y_train = encode_one_hot(labels_list, (np.max(labels_list) + 1))


    #Make sure the shapes we have of the train and test sets are correct
    assert X_train.shape == (50000, 3072)
    assert Y_train.shape == (50000, 10)
    assert X_test.shape == (10000, 3072)
    assert Y_test.shape == (10000, 10)

    return X_train.T, Y_train.T, X_test.T, Y_test.T

def initialize_parameters(layer_dims):
    """
        Initialized the parameters for the neural network
    """
    parameters = {}
    num_layers = len(layer_dims)
    np.random.seed(1)

    for l in range(1, num_layers):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

        assert parameters["W"+str(l)].shape == (layer_dims[l], layer_dims[l - 1])
        assert parameters["b"+str(l)].shape == (layer_dims[l], 1)

    return parameters


def one_layer_forward(W, b, act_prev):
    Z = np.dot(W, act_prev) + b
    assert Z.shape == (W.shape[0], act_prev.shape[1])

    return Z, (act_prev, W, b)

def one_layer_activation(W, b, act_prev, activation_type):
    Z, lin_cache = one_layer_forward(W, b, act_prev)
    if activation_type == 'relu':
        activation, act_cache = relu_activation(Z)
    if activation_type == 'sigmoid':
        activation, act_cache = sigmoid_activation(Z)

    return activation, (lin_cache, act_cache)

def forward_propagation(X_train, parameters):
    act_prev = X_train
    num_layers = len(parameters) // 2
    caches = []

    for l in range(1, num_layers):
        A, cache = one_layer_activation(parameters["W"+str(l)], parameters["b"+str(l)], act_prev, "relu")
        caches.append(cache)
        act_prev = A
    
    A_final, cache = one_layer_activation(parameters["W"+str(num_layers)], parameters["b"+str(num_layers)], act_prev, "sigmoid")
    caches.append(cache)
    assert A_final.shape == (10, X_train.shape[1])

    return A_final, caches

def get_cost(A_final, Y_train):
    m = Y_train.shape[1]

    cost = (-1/m) * np.sum(np.multiply(Y_train, np.log(A_final)) + np.multiply((1 - Y_train), np.log(1 - A_final)))
    cost = np.squeeze(cost)
    
    assert cost.shape == ()

    return cost


def one_layer_backward(dZ, cache):
    act_prev, W, b = cache
    m = act_prev.shape[1]

    dW = (1/m) * np.dot(dZ, act_prev.T)
    db = (1/m) * np.sum(dZ, keepdims=True, axis=1)
    dA_prev = np.dot(W.T, dZ)

    assert dA_prev.shape == act_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    return dW, db, dA_prev


def activation_backward(dA, cache, activation_type):
    lin_cache, act_cache = cache
    if activation_type == 'relu':
        dZ = relu_backward(dA, act_cache)
    elif activation_type == 'sigmoid':
        dZ = sigmoid_backward(dA, act_cache)

    dW, db, dA_prev = one_layer_backward(dA, lin_cache)

    return dW, db, dA_prev


def backward_propagation(A_final, Y_train, caches):
    num_layers = len(caches)
    gradients = {}
    m = A_final.shape[1]
    Y_train = Y_train.reshape(A_final.shape)

    dA_final = - (np.divide(Y_train, A_final) - np.divide((1 - Y_train), (1 - A_final)))

    current_cache = caches[num_layers - 1]
    a, b, c = activation_backward(dA_final, current_cache, 'sigmoid')
    gradients["dW"+str(num_layers)] = a
    gradients["db"+str(num_layers)] = b
    gradients["dA"+str(num_layers)] = c

    for l in reversed(range(num_layers - 1)):
        current_cache = caches[l]
        dA_prev = gradients["dA"+str(l+2)]
        a, b, c = activation_backward(dA_prev, current_cache, 'relu')
        gradients["dW" + str(l + 1)] = a
        gradients["db" + str(l + 1)] = b
        gradients["dA" + str(l + 1)] = c

    return gradients


def update_parameters(parameters, gradients, learning_rate=0.05):
    num_layers = len(parameters) // 2

    for l in range(num_layers):
        parameters["W"+str(l + 1)] -= learning_rate * gradients["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * gradients["db" + str(l + 1)]

    return parameters


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = get_data()
    paramters = initialize_parameters([X_train.shape[0], 128, 128, 128, 128, 10])
    num_iterations = 101

    for i in range(num_iterations):
        print('Iteration ' + str(i))
        A_final, caches = forward_propagation(X_train, paramters)
        print(get_cost(A_final, Y_train))
        grads = backward_propagation(A_final, Y_train, caches)
        parameters = update_parameters(paramters, grads)
