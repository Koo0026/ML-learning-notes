import numpy as np
import matplotlib.pylab as plt

# 나쁜 미분 구현 예시


def numerical_diff(f, x):
    # rounding error문제를 일으킴.
    # 접선의 기울기가아닌 두점사이의 기울기를 구하는 두번째 문제.
    h = 1e-50
    return (f(x+h) - f(x)) / (h)


def numerical_diff2(f, x):
    h = 1e-4  # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x


def function_2(x):
    # 이때 x는 넘파이배열.
    return x[0]**2 + x[1]**2

# 편미분


def function_2_tmp1(x0):
    # x1이 4일때
    return x0*x0 + 4.0**2.0


def function_2_tmp2(x1):
    # x0이 3일때
    return 3.0**2.0 + x1*x1


def numerical_gradient(f, x):
    h = 1e-4  # 0.001
    grad = np.zeros_like(x)
    # x와 형상이 같은 0배열 생성. 이때는 x가 1차원 배열이므로 grad도 1차원 배열이 된다.

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val  # 값 복원

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    # f는 최적화하려는 함수, init_x는 초기값, lr은 학습률, step_num은 반복횟수를 의미함.

    for i in range(step_num):
        print(f"{i}번쨰 학습중 ")
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))
