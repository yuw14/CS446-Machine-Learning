def cross_entropy(Y, Y_hat):
    """
    @brief    Compute cross-entropy loss between labels and predictions averaged across the N instances
    @param    Y       ground-truth label of shape (N x T x K)
    @param    Y_hat   predictions of shape (N x T x K)
    @return   the average cross-entropy loss between Y and Y_hat
    """
    (n, T, _) = Y.shape
    total_loss = 0
    for bi in range(n):
        for t in range(T):
            total_loss += -np.dot(Y[bi, t], np.log(Y_hat[bi, t]+1e-9))
    return total_loss / n


def generate_labels(X):
    """
    @brief      Takes in samples and generates labels
    @param      X Samples of sequence data (N x T x D)
    @return     Y, labels of shape (N x T x K)
    """
    (n, t, d) = X.shape
    Y = np.zeros((n, t, d+1))
    Y[:, :-1, :-1] = X[:, 1:]
    Y[:, -1, d] = 1
    return Y


def softmax_batch(z):
    y_hat = np.empty_like(z)
    for i in range(z.shape[0]):
        y_hat[i] = softmax(z[i])
    return y_hat


class RNNCell(Module):
    def __init__(self, parameters):
        super().__init__()
        self._register_param('V', parameters['V'])
        self._register_param('W', parameters['W'])
        self._register_param('U', parameters['U'])
        self._register_param('c', parameters['c'])
        self._register_param('b', parameters['b'])

    def forward(self, x_t, h_prev):
        """
        @brief      Takes a batch of input at the timestep t with the previous hidden states and compute the RNNCell output
        @param      x_t    A numpy array as the input (N, in_features)
        @param      h_prev A numpy array as the previous hidden state (N, hidden_features)
        @return     The current hidden state (N, hidden_features) and the prediction at the timestep t (N, out_features)
        """
        # IMPLEMENT ME
        h_t = np.tanh(np.dot(
            h_prev, self.params['W'].T) + np.dot(x_t, self.params['U'].T) + self.params['b'])
        y_hat_t = softmax_helper(
            np.dot(h_t, self.params['V'].T) + self.params['c'])
        return h_t, y_hat_t

    def backward(self, x_t, y_t, y_hat_t, dh_next, h_t, h_prev):
        """
        @brief      Compute and update the gradients for parameters of RNNCell at the timestep t
        @param      x_t      A numpy array as the input (N, in_features)
        @param      y_t      A numpy array as the target (N, out_features)
        @param      y_hat_t  A numpy array as the prediction (N, out_features)
        @param      dh_next  A numpy array as the gradient of the next hidden state (N, hidden_features)
        @param      h_t      A numpy array as the current hidden state (N, hidden_features)
        @param      h_prev   A numpy array as the previous hidden state (N, hidden_features)
        @return     The gradient of the current hidden state (N, hidden_features)
        """
        # IMPLEMENT ME
        N = x_t.shape[0]
        H = h_t.shape[1]
        DELTA = np.zeros((N, H))

        for n in range(N):
            diff_y = y_hat_t[n] - y_t[n]
            delta = np.zeros(H)
            if (dh_next == np.zeros_like(dh_next)).all():  # k=t
                delta = (1-h_t[n]**2)*np.dot(self.params['V'].T, diff_y)
            else:  # k < t
                delta = (1-h_t[n]**2) * (np.dot(self.params['W'].T,
                                                dh_next[n]) + np.dot(self.params['V'].T, diff_y))
            DELTA[n] = delta
            self.grads['V'] += np.outer(diff_y, h_t[n])
            self.grads['c'] += diff_y
            self.grads['U'] += np.outer(delta, x_t[n])
            self.grads['b'] += delta
            self.grads['W'] += np.outer(delta, h_prev[n])

        return DELTA


class RNN(Module):
    def __init__(self, d, h, k):
        """
        @brief      Initialize weight and bias
        @param      d   size of the input layer
        @param      h   size of the hidden layer
        @param      k   size of the output layer
        NOTE: Do not change this function or variable names; they are
            used for grading.
        """
        super().__init__()
        self.d = d
        self.h = h
        self.k = k

        parameters = {}
        wb = weight_init(d + h + 1, h)
        parameters['W'] = wb[:, :h]
        parameters['U'] = wb[:, h:h+d]
        parameters['b'] = wb[:, h+d]

        wb = weight_init(h + 1, k)
        parameters['V'] = wb[:, :h]
        parameters['c'] = wb[:, h]
        self._register_child('RNNCell', RNNCell(parameters))

    def forward(self, X):
        """
        @brief      Takes a batch of samples and computes the RNN output
        @param      X   A numpy array as the input of shape (N x T x D)
        @return     Hidden states (N x T x H), RNN's output (N x T x K)
        """
        # IMPLEMENT ME
        K = self.children['RNNCell'].params['V'].shape[0]
        H = self.children['RNNCell'].params['V'].shape[1]
        N = X.shape[0]
        T = X.shape[1]
        D = X.shape[2]
        hs = np.zeros((N, T, H))
        y_hat = np.zeros((N, T, K))
        ht = np.zeros((N, H))
        for t in range(T):
            ht, y_hat_t = self.children['RNNCell'].forward(X[:, t, :], ht)
            hs[:, t, :] = ht
            y_hat[:, t, :] = y_hat_t
        return hs, y_hat

    def backward(self, X, Y, Y_hat, H):
        """
        @brief      Backpropagation of the RNN
        @param      X      A numpy array as the input of shape (N x T x D)
        @param      Y      A numpy array as the ground truth labels of shape (N x T x K)
        @param      Y_hat  A numpy array as the prediction of shape (N x T x K)
        @param      H      A numpy array as the hidden states after the forward of shape (N x T x H)
        """
        # IMPLEMENT ME
        T = X.shape[1]
        N = H.shape[0]
        scale_H = H.shape[2]
        dh_next = np.zeros((N, scale_H))
        for t in range(T-1, -1, -1):
            if t == 0:
                h_prev = np.zeros((N, scale_H))
            else:
                h_prev = H[:, t-1, :]
            dh_next = self.children['RNNCell'].backward(
                X[:, t, :], Y[:, t, :], Y_hat[:, t, :], dh_next, H[:, t, :], h_prev)


def train_one_epoch(model, X, test_X, lr):
    """
    @brief      Takes in a model and train it for one epoch.
    @param      model   The recurrent neural network
    @param      X       The features of training data (N x T x D)
    @param      test_X  The features of testing data (M x T x D)
    @param      lr      Learning rate
    @return     (train_cross_entropy, test_cross_entropy), the cross
                entropy loss for train and test data
    """
    # IMPLEMENT ME
    clear_grad(model)
    Y = generate_labels(X)
    Hideen_States, Y_hat = model.forward(X)
    train_cross_entropy = cross_entropy(Y, Y_hat)

    model.backward(X, Y, Y_hat, Hideen_States)

    N = X.shape[0]
    step_size = lr/N
    model.children['RNNCell'].params['W'] -= step_size * \
        model.children['RNNCell'].grads['W']
    model.children['RNNCell'].params['U'] -= step_size * \
        model.children['RNNCell'].grads['U']
    model.children['RNNCell'].params['b'] -= step_size * \
        model.children['RNNCell'].grads['b']
    model.children['RNNCell'].params['V'] -= step_size * \
        model.children['RNNCell'].grads['V']
    model.children['RNNCell'].params['c'] -= step_size * \
        model.children['RNNCell'].grads['c']

    test_Y = generate_labels(test_X)
    new_H, testY_hat = model.forward(test_X)
    test_cross_entropy = cross_entropy(test_Y, testY_hat)

    return train_cross_entropy, test_cross_entropy
