import numpy as np
from activation_function import softmax_notoverfolow
from derivation import numerical_gradient
from loss_funtion import cross_entropy_error

class simpleNet:
  def __init__(self):
    self.W = np.random.randn(2,3) # 정규분포를 따르는 난수로 초기화

  def predict(self, x):
    return np.dot(x, self.W)
  
  def loss(self, x, t):
    z = self.predict(x)
    y = softmax_notoverfolow(z)
    loss = cross_entropy_error(y, t)

    return loss
    

net = simpleNet()
print(net.W)
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p)) # 최댓값의 인덱스 반환

t = np.array([0, 0, 1]) # 정답 레이블
print(net.loss(x, t)) # 손실함수의 값 출력

def f(W):
  return net.loss(x, t)   

dW = numerical_gradient(f, net.W)
print(dW)