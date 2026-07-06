import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    # step function은 실수(부동소수점)만 받아들임. 즉 넘파이 배열을 인수로 넣을 수는 없음.
    if x > 0:
        return 1
    else:
        return 0


def step_function_np(x):
    # step function을 넘파이 배열로 구현
    y = x > 0  # array(True, False, True, ...) 형태로 반환
    return y.astype(np.int)

# numpy에서는 astype() 메서드를 사용하여 자료형을 변환할 수 있음.


def step_function_np2(x):
    return np.array(x > 0, dtype=int)


def sigmoid(x):
    # broadcasting을 이용하여 넘파이 배열도 인수로 넣을 수 있음.
    return 1/(1+np.exp(-x))


def relu(x):
    return np.maximum(0, x)
