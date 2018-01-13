import numpy as np


def encode_one_hot(targets, num_classes):
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

    #One hot encoding to get correct labels matrix
    labels_list = np.concatenate(labels_list)
    Y_train = encode_one_hot(labels_list, (np.max(labels_list) + 1))


    #Make sure the shapes we have of the train and test sets are correct
    assert(X_train.shape == (50000, 3072))
    assert(Y_train.shape == (50000, 10))
    assert(X_test.shape == (10000, 3072))
    assert(Y_test.shape == (10000, 10))

    return X_train, Y_train, X_test, Y_test




if __name__ == '__main__':
    get_data()