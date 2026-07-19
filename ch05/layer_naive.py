class MulLayer:
  #곱셉계층 
  def __init__(self):
    self.x = None
    self.y = None

  def forward(self, x, y):
    #순전파 
    self.x = x
    self.y = y
    out = x * y 

    return out
  
  def backward(self, dout):
    #역전파 
    #상류에서 넘어온 미분(dout)에 순전파때의 값을 서로 '바꿔' 곱함. 
    dx = dout * self.y
    dy = dout * self.x 

    return dx, dy
  
class AddLayer:
  #덧셈계층
  def __init__(self):
    #역전파 과정에서 원래의 순전파에서의 입력값이 필요없으므로 초기화도 필요하지않음.
    pass

  def forward(self, x, y):
    out = x + y 
    return out
    
  def backward(self, dout):
    dx = dout * 1
    dy = dout * 1
    return dx, dy 


