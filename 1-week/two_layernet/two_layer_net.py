from common.function import *
from common.gradient import numerical_gradient
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # input_size, hidden_size, output_size 순서대로 입력층, 은닉층, 출력층의 뉴런 수를 의미함.

        # params: 매개변수 저장하는 dictionary 변수
        # bias는 처음에 0 으로 시작해도 되므로 np.zeros를 사용함.
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        # 추론을 수행.
        W1, W2 = self.params['W1'], self.params["W2"]
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        # 손실함수의 값을 구함.
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        # 정확도를 구함.
        y = self.predict(x)
        # 한 행마다 최대확률 값의 index를 가져와서 비교.
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        # 가중치 매개변수의 기울기를 구함.
        def loss_W(W): return self.loss(x, t)

        # grads: 기울기 저장하는 dictionary 변수
        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params['W1'])
        grads["b1"] = numerical_gradient(loss_W, self.params['b1'])
        grads["W2"] = numerical_gradient(loss_W, self.params['W2'])
        grads["b2"] = numerical_gradient(loss_W, self.params['b2'])

        return grads
