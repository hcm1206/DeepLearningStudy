# coding: utf-8
import sys
sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *

# 계산할 함수식 정의
def f(x, y):
    return x**2 / 20.0 + y**2

# 계산할 함수의 미분(기울기) 식 정의 (그냥 해석적 수치 미분)
def df(x, y):
    return x / 10.0, 2.0*y

# 초기 좌표(매개변수) 입력값
init_pos = (-7.0, 2.0)
# 매개변수를 저장할 딕셔너리
params = {}
# 초기 매개변수로 최초 값(init_pos) 지정
params['x'], params['y'] = init_pos[0], init_pos[1]
# 기울기를 저장할 딕셔너리
grads = {}
# 'x'와 'y' 매개변수의 최초 기울기 값으로 0,0 지정
grads['x'], grads['y'] = 0, 0

# 최적화 방법(클래스)를 순서가 존재하는 딕셔너리 자료구조로 저장
optimizers = OrderedDict()
# 학습률 0.95의 'SGD' 최척화 기법을 최적화 딕셔너리에 저장
optimizers["SGD"] = SGD(lr=0.95)
# 학습률 0.1의 '모멘텀' 최적화 기법을 최적화 딕셔너리에 저장
optimizers["Momentum"] = Momentum(lr=0.1)
# 학습률 1.5의 "AdaGrad" 최적화 기법을 최적화 딕셔너리에 저장
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
# 학습률 0.3의 "Adam" 최적화 기법을 최적화 딕셔너리에 저장
optimizers["Adam"] = Adam(lr=0.3)

# 최초 인덱스를 1로 지정
idx = 1

# 최적화 기법을 차례로 입력하며 반복
for key in optimizers:
    # 현재 최적화 기법을 [key]의 최적화로 설정
    optimizer = optimizers[key]
    # 계산되었던 x값들을 담을 리스트 설정
    x_history = []
    # 계산되었던 y값들을 담을 리스트 설정
    y_history = []
    # x와 y 매개변수의 초기값 지정(-7.0, 2.0)
    params['x'], params['y'] = init_pos[0], init_pos[1]
    
    # 30번 반복
    for i in range(30):
        # 현재 매개변수 x의 값을 리스트에 원소로 추가
        x_history.append(params['x'])
        # 현재 매개변수 y의 값을 리스트에 원소로 추가
        y_history.append(params['y'])
        
        # x와 y의 기울기를 미분식을 통해 구함
        grads['x'], grads['y'] = df(params['x'], params['y'])
        # 구한 기울기를 이용해 매개변수 갱신
        optimizer.update(params, grads)
    
    # x 값을 -10부터 10 미만의 0.01 간격 넘파이 배열로 저장
    x = np.arange(-10, 10, 0.01)
    # y 값을 -5부터 5 미만의 0.01 간격 넘파이 배열로 저장
    y = np.arange(-5, 5, 0.01)
    
    # x축을 x 배열, y축을 y 배열로 하는 격자 위치의 x축 좌표, y축 좌표를 각각 X, Y에 배열로 저장
    X, Y = np.meshgrid(x, y) 
    # Z에 X, Y를 계산할 함수에 넣은 결과값 저장
    Z = f(X, Y)
    
    # 외곽선 단순화하기
    mask = Z > 7
    Z[mask] = 0
    
    # 그래프 그리기
    plt.subplot(2, 2, idx)
    idx += 1
    plt.plot(x_history, y_history, 'o-', color="red")
    plt.contour(X, Y, Z)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    #colorbar()
    #spring()
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")
    
plt.show()
