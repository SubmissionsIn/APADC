import os, sys
import numpy as np
import scipy.io as sio


def load_data(config):
    """Load data """
    data_name = config['dataset']
    main_dir = sys.path[0]
    X_list = []
    Y_list = []
    print("shuffle")
    if data_name in ['MNIST-USPS']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        X_list.append(mat['X1'].astype('float32'))          # (5000,784)
        X_list.append(mat['X2'].astype('float32'))          # (5000,784)
        Y_list.append(np.squeeze(mat['Y']))
        print(Y_list[0])
    elif data_name in ['Caltech101-20']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        X = mat['X'][0]
        # for view in [3, 4]:
        #     x = X[view]                                     # (2386,1984)
        #     X_list.append(x)                                # (2386,512)
        # y = np.squeeze(mat['Y']).astype('int')
        # Y_list.append(y)
        print(np.squeeze(mat['Y']).astype('int'))

        x1 = X[3]
        x2 = X[4]
        xx1 = np.copy(x1)
        xx2 = np.copy(x2)
        Y = np.copy(mat['Y'])
        index = [i for i in range(2386)]
        np.random.seed(2386)
        np.random.shuffle(index)
        for i in range(2386):
            xx1[i] = x1[index[i]]
            xx2[i] = x2[index[i]]
            Y[i] = mat['Y'][index[i]]

        X_list.append(xx1)
        X_list.append(xx2)
        y = np.squeeze(Y).astype('int')
        Y_list.append(y)
        print(y)
    elif data_name in ['RGB-D']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'RGB-D.mat'))
        x2 = np.copy(mat['X2'])
        x1 = np.copy(mat['X1'])
        y = np.copy(mat['Y'].T)
        print(np.squeeze(y))
        np.random.seed(1449)
        index = [i for i in range(1449)]
        np.random.shuffle(index)
        for i in range(1449):
            x2[i] = mat['X2'][index[i]]
            x1[i] = mat['X1'][index[i]]
            y[i] = mat['Y'].T[index[i]]

        from sklearn.preprocessing import normalize
        x1 = normalize(x1, axis=1, norm='max')
        x2 = normalize(x2, axis=1, norm='max')
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        x1 = min_max_scaler.fit_transform(x1)
        x2 = min_max_scaler.fit_transform(x2)

        X_list.append(x2.astype('float32'))                 # (1449,2048)
        X_list.append(x1.astype('float32'))                 # (1449,300)
        Y_list.append(np.squeeze(y))
        print(np.squeeze(y))
    elif data_name in ['Scene-15']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'Scene-15.mat'))
        X = mat['X'][0]
        # X_list.append(X[1].astype('float32'))              # (4485,59)
        # X_list.append(X[0].astype('float32'))              # (4485,20)
        # Y_list.append(np.squeeze(mat['Y']))
        print(np.squeeze(mat['Y']))

        x1 = X[1]
        x2 = X[0]
        xx1 = np.copy(x1)
        xx2 = np.copy(x2)
        Y = np.copy(mat['Y'])
        index = [i for i in range(4485)]
        np.random.seed(4485)
        np.random.shuffle(index)
        for i in range(4485):
            xx1[i] = x1[index[i]]
            xx2[i] = x2[index[i]]
            Y[i] = mat['Y'][index[i]]

        # from sklearn.preprocessing import normalize
        # xx1 = normalize(xx1, axis=1, norm='max')
        # xx2 = normalize(xx2, axis=1, norm='max')
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        xx1 = min_max_scaler.fit_transform(xx1)
        xx2 = min_max_scaler.fit_transform(xx2)

        X_list.append(xx1)
        X_list.append(xx2)
        y = np.squeeze(Y).astype('int')
        Y_list.append(y)
        print(y)
    elif data_name in ['NoisyMNIST']:
        data = sio.loadmat('./data/NoisyMNIST.mat')
        train = DataSet_NoisyMNIST(data['X1'], data['X2'], data['trainLabel'])
        tune = DataSet_NoisyMNIST(data['XV1'], data['XV2'], data['tuneLabel'])
        test = DataSet_NoisyMNIST(data['XTe1'], data['XTe2'], data['testLabel'])
        # X_list.append(np.concatenate([tune.images1, test.images1], axis=0))
        # X_list.append(np.concatenate([tune.images2, test.images2], axis=0))
        # Y_list.append(np.concatenate([np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])]))
        X_list.append(np.concatenate([train.images1, tune.images1, test.images1], axis=0))
        X_list.append(np.concatenate([train.images2, tune.images2, test.images2], axis=0))
        Y_list.append(np.concatenate([np.squeeze(train.labels[:, 0]), np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])]))
        print(Y_list[0])
        x1 = X_list[0]
        x2 = X_list[1]
        xx1 = np.copy(x1)
        xx2 = np.copy(x2)
        Y = np.copy(Y_list[0])
        index = [i for i in range(70000)]
        np.random.seed(784)
        np.random.shuffle(index)
        for i in range(70000):
            xx1[i] = x1[index[i]]                    # (70000, 784)
            xx2[i] = x2[index[i]]                    # (70000, 784)
            Y[i] = Y_list[0][index[i]]
        print(Y)
        X_list = [xx1, xx2]
        Y_list = [Y]
    return X_list, Y_list


class DataSet_NoisyMNIST(object):
    def __init__(self, images1, images2, labels, fake_data=False, one_hot=False,
                 dtype=np.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        if dtype not in (np.uint8, np.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images1.shape[0] == labels.shape[0], (
                    'images1.shape: %s labels.shape: %s' % (images1.shape,
                                                            labels.shape))
            assert images2.shape[0] == labels.shape[0], (
                    'images2.shape: %s labels.shape: %s' % (images2.shape,
                                                            labels.shape))
            self._num_examples = images1.shape[0]

            if dtype == np.float32 and images1.dtype != np.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                # print("type conversion view 1")
                images1 = images1.astype(np.float32)

            if dtype == np.float32 and images2.dtype != np.float32:
                # print("type conversion view 2")
                images2 = images2.astype(np.float32)

        self._images1 = images1
        self._images2 = images2
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images1(self):
        return self._images1

    @property
    def images2(self):
        return self._images2

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [fake_image for _ in range(batch_size)], [fake_label for _
                                                                                                      in range(
                    batch_size)]

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images1 = self._images1[perm]
            self._images2 = self._images2[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._images1[start:end], self._images2[start:end], self._labels[start:end]
