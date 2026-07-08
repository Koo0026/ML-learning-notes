from exercise.activation_function import sigmoid, softmax_notoverfolow
from mnist import load_mnist
import pickle
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def get_data():
    (x_tran, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False)
    # normalize는 전처리작업으로 정규화를 수행하는 옵션이다. True로 설정하면 0~255의 픽셀값을 0.0~1.0 사이의 값으로 정규화한다.

    return x_test, t_test


def init_network():
    curr_dir = os.path.dirname(__file__)

    file_path = os.path.join(curr_dir, "sample_weight.pkl")
    with open(file_path, 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax_notoverfolow(a3)

    return y


x, t = get_data()
network = init_network()

# accuracy_cnt = 0
# for i in range(len(x)):
#  y = predict(network, x[i])
#  p = np.argmax(y)
#  if p == t[i]:
# accuracy_cnt += 1
# x 하나하나가 28* 28 의 이미지를 flatten한 784차원의 벡터이므로, predict() 함수에 x[i]를 넣어주면 된다.


# batch processing
batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
