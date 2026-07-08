import numpy as np
from mnist import load_mnist
#mnist 데이터셋은 0~9까지의 숫자 이미지 데이터셋이다.
#mnist.py 파일은 이미지를 넘파이배열로 변환하는 기능을 제공한다.
from PIL import Image


def img_show(img):
    #numpy로 저장된 이미지 데이터를 PIL용 데이터 객체로 변환. 
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
#load_mnist 함수는 MNIST 데이터를 (훈련이미지, 훈련레이블), (시험이미지, 시험레이블) 형식으로 반환한다. 

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 원래 이미지 크기인 28x28로 변형
print(img.shape)  # (28, 28)
img_show(img)
