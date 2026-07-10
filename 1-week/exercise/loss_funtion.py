import numpy as np


def sum_squared_error(y, t):
    # y는 신경망의 출력, t는 정답 레이블이다.
    # 이때의 t값은 one-hot encoding으로 표현된 벡터이다.
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    # delta는 log(0)으로 인해 발생하는 수치적 불안정성을 방지하기 위해 추가된 작은 값이다.
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def cross_entropy_error_batch_onehotencoding(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def cross_entropy_error_batch_number_label(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # y([np.arange(batch_size), t])는 y의 각 행에서 정답 레이블에 해당하는 원소를 선택한다.(확률값)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
