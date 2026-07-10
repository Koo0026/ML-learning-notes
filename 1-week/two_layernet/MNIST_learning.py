from training_image.mnist import load_mnist
import numpy as np
from two_layer_net import TwoLayerNet
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True)
