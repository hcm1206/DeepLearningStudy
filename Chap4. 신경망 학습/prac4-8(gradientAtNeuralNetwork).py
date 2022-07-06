# 신경망 학습에서 가중치 매개변수에 대한 손실함수의 기울기를 구해야 함
# 각각의 가중치에 대한 편미분(기울기) 값을 계산하여 가중치 행렬과 형상이 같은 편미분값(기울기) 행렬을 구할 수 있음

import sys, os
sys.path.append(os.pardir)
import numpy as np

# 소프트맥스 함수(활성화 함수)
def softmax(a):
    c = max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y

# 교차 엔트로피 오차 함수(손실 함수)
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y+delta))

# 편미분 함수(기울기 함수)
def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) 
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val
        
    return grad

# 배치가 존재하는 다차원 배열도 계산 가능한 편미분 구하는 함수
def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        return grad

# 간단한 신경망 클래스
class simpleNet:
    def __init__(self):
        # 랜덤으로 저장된 2X3 배열의 가중치 매개변수 배열을 인스턴스 변수 W에 저장
        self.W = np.random.randn(2,3)

    # 입력값 x값를 이용하여 예측을 수행하는 함수
    def predict(self,x):
        # 입력값 배열 x와 가중치 배열 self.W를 행렬곱하여 리턴
        return np.dot(x,self.W)

    # 입력값 배열 x, 정답 배열 t를 입력받아 손실함수 값을 구하는 함수
    def loss(self, x, t):
        # z에 입력 값을 가중치와 곱하여 출력받은 예측값 저장
        z = self.predict(x)
        # y에 소프트맥스 함수(활성화 함수)에 예측값 z를 입력하여 나온 출력값 저장
        y = softmax(z)
        # loss에 교차 엔트로피 오차 함수(손실 함수)에 계산값 y와 정답값 t를 입력하여 나온 출력값 저장
        loss = cross_entropy_error(y,t)
        # 최종 출력값 loss 리턴
        return loss

# net에 simpleNet 클래스 객체(임의로  구현된 단순한 신경망) 저장
net = simpleNet()
# net 신경망의 가중치(2×3 형상의 행렬에 랜덤 값으로 생성되어 있음) 출력
print(net.W)
# 입력값으로 [0.6, 0.9] 배열 지정
x = np.array([0.6,0.9])
# net 신경망 매개변수와 입력값을 행렬곱한 계산값을 p에 저장
p = net.predict(x)
# p 출력
print(p)
# 계산값 배열 중에서 최댓값인 원소 인덱스 출력
print(np.argmax(p))
# 실제 정답 레이블 [0,0,1] 배열(2번 인덱스가 정답)을 t에 저장
t = np.array([0,0,1])
# x와 t의 값을 비교하여 손실함수 값을 계산 후 출력
print(net.loss(x, t))


# 편미분을 진행할 손실함수를 f 함수로 정의 (W 인수는 의미가 없지만 일관성을 위해 넣어둠)
def f(W):
    return net.loss(x,t)
# 위 방법 대신 f = lambda W: net.loss(x,t) 와 같이 람다식으로 써도 됨

# dW에 f 함수(전체 손실 함수)에 가중치(net.W) 값에서의 편미분 값 저장
dW = numerical_gradient(f, net.W)
print(dW)
print(dW[1][1])

# dW에서 나온 값을 토대로 전체 손실 함수(loss)의 최소값을 만들도록 하는 가중치 값을 찾아나갈 수 있음
# ex1. dW[0][0]이 약 0.2라면 net.W[0][0]을 h만큼 늘리면 전체 손실 함수(loss)의 출력값이 0.2h만큼 증가한다는 뜻
# ex2. dW[1][2]가 약 -0.5라면 net.W[1][2]를 h만큼 늘리면 전체 손실 함수(loss)의 출력값이 0.5h만큼 감소한다는 뜻