# ML-learning-notes

이번 길지 않는 공부할 딥러닝에 대해 이 페이지에서 앞으로 써 내려갈 예정이다.
먼저 python을 이용할 것이며, numpy와 matplotlib에 관한 기본적인 지식들을 알고 있는 가정으로 써내려간다.

첫번째로, 신경망에 대해서 배우기전에 percepticon에 대해서 알아보자.
percepticon은 여러개의 입력으로 하나의 output을 산출한다. 

가장 간단한 예시로 풀어나가 보자. 
두개의 입력신호($x_1,  x_2$)가 존재한다고 가정하였을때, 각각의 입력신호는 고유한 weight 값과 곱해진다.
이때 이 weight들끼리의 합이 임계값($\theta$)를 초과하였을때 1을 output으로 가지며 그렇지않으면 0을 output으로 가진다.
이를 수식으로 정리하면 

$$
y = \begin{cases}
0 & (w_1x_1 + w_2x_2 \le \theta) \\
1 & (w_1x_1 + w_2x_2 > \theta)
\end{cases}
$$

이때의 theta를 -b로 치환하여 일반화를 시키면 아래의 식을 도출해낼 수 있다. 
이떄의 b는 bias를 의미하며 얼마나 쉽게 뉴런이 활성화될지를 결정한다.

$$
y = \begin{cases}
0 & (b+ w_1x_1 + w_2x_2 \le 0) \\
1 & (b + w_1x_1 + w_2x_2 > 0)
\end{cases}
$$

이러한 percepticon으로 간단한 AND, NAND, OR게이트를 구현해낼 수 있다. 
각각의 GATE들에 대한 python코드는 1-week의 percepticon.py에서 확인할 수 있다.

하지만 이러한 단층 percepticon은 선형으로 XOR은 구현해내지 못 한다.
그리하여, 비선형영역인 multi-layer percepticon 개념을 도입하게된다. 
이 multi-layer percepticon을 이용하여 XOR게이트를 구현해 낼 수 있으며, 이론상 모든 컴퓨터구조를 구현해 낼 수 있다.

여기까지 percepticon에 대해 알아보았다. 
이 percepticon은 여전히 사람이 weight와 bias를 선택해야한다. 하지만 이를 기계가 학습하여 선택하는 것을 우리는 목표로 하기에 
앞으로 신경망에 대해서 알아보도록하자. 
