# 신경망 학습의 목적은 손실함수의 값을 최소한으로 하는 매개변수를 찾는 것 : 최적화(Optimization)
# 지금까지 구현해온 신경망 학습 방법은 매개변수의 기울기를 통해 매개변수가 기울어진 방향을 따라 일정 값만큼 이동하며 매개변수를 갱신하는 방식
# 이를 확률적 경사 하강법(SGD)라고 정의

# 확률적 경사 하강법(SGD) 클래스 구현
class SGD:
    # 클래스 생성시 학습률을 입력받음(기본값 0.01)
    def __init__(self, lr=0.01):
        # 입력받은 학습률 변수를 클래스 변수로 저장
        self.lr = lr

    # 각 매개변수(딕셔너리)와 매개변수의 기울기(딕셔너리)를 입력받아 SGD 방식으로 매개변수 갱신
    def update(self, params, grads):
        # 매개변수가 저장된 딕셔너리를 통해 각 매개변수 반복
        for key in params.keys():
            # 해당 매개변수 값을 해당 매개변수의 기울기와 학습률을 곱한 값으로 빼서 저장
            params[key] -= self.lr * grads[key]


# 이러한 SGD 클래스를 정의했다면 매개변수 갱신(optimizer)의 역할은 SGD 클래스가 맡게 됨
# 이처럼 최적화를 담당하는 클래스를 분리하여 구현하면 기능 모듈화하기 좋음


# SGD의 단점 : 비등방성 함수(anisotropy)에서는 탐색 경로가 비효율적
# 비등방성 함수 : 방향에 따라 기울기가 달라지는 함수
# 1/20*x**2 + y**2 함수의 경우 최소값이 되는 장소는 (0,0)이지만 대다수의 매개변수 기울기는 (0,0) 방향을 가리키지 않고
# 그저 y = 0인 직선방향으로만 가리킴

# 예시) 기울기 그래프

import numpy as np
import matplotlib.pylab as plt

# 편미분 함수
def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val
        
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad

# 1/20*x**2 + y**2 함수
def function_1(x):
    return 1/20*x[0]**2 + x[1]**2


if __name__ == '__main__':
    x = np.arange(-10, 11, 1)
    y = np.arange(-5,6,1)
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    grad = numerical_gradient(function_1, np.array([X, Y]).T).T


    plt.figure()
    plt.quiver(X, Y-0.1, -grad[0], -grad[1],  angles="xy",color="#666666")
    plt.xlim([-10, 10])
    plt.ylim([-5, 5])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.draw()
    plt.show()