from layer_naive import MulLayer, AddLayer

#사과쇼핑 예시

apple = 100
apple_num = 2
tax = 1.1

#사과 2개 구입

#계층들 정의: 계층은 신경망의 기능단위를 의미함.
mul_apple_layer = MulLayer()
mul_tex_layer = MulLayer()

#순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tex_layer.forward(apple_price, tax)
print(price)

#역전파
dprice = 1
dapple_price, dtax = mul_tex_layer.backward(dprice)
dapple, dappple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dappple_num, dtax) 
