import matplotlib.pyplot as plt
from mnist import load_mnist
import numpy as np
from two_layer_net import TwoLayerNet


(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 1
train_size = x_train.shape[0]  # 60000
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list= []
test_acc_list =[]

iter_per_epoch = max(train_size / batch_size, 1)
#1epoch당 반복 수

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    #과대적합을 일으키지 않은지 확인 필요. 
    #1epoch별로 훈련 데이터와 시험 데이터를 대상으로 정확도 기록. 
    if i % iter_per_epoch == 0: 
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc ㅣ" + str(train_acc) + ", " + str(test_acc))

x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title("Training Loss")
plt.show()

