import numpy as np
from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

#print(x_train.shape)  # (60000, 784)
#print(t_train.shape)  # (60000, 10)

#무작위로 10장 추출
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
#0~59999 사이의 숫자 중에서 10개를 무작위로 선택하여 batch_mask에 저장한다.
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

