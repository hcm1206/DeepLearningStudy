# 3층 신경망 구현 정리

import numpy as np

# 활성화 함수(시그모이드 함수) 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
# 항등 함수(코드 구조 일관성 목적) 정의
def identify_function(x):
    return x

# 신경망을 임의로 정의(가중치, 편향를 임의로 정의)하여 딕셔너리 형태로 리턴하는 함수
def init_network():
    network = {}
    network["W1"] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network["b1"] = np.array([0.1,0.2,0.3])
    network["W2"] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network["b2"] = np.array([0.1,0.2])
    network["W3"] = np.array([[0.1,0.3],[0.2,0.4]])
    network["b3"] = np.array([0.1,0.2])
    return network

# 신경망과 입력값 배열을 입력받아 순방향 처리하여 출력값 리턴하는 함수
def forward(network, x):
    # 신경망에서 각각의 가중치와 편향을 추출
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    # 입력 값을 신경망 과정을 거쳐 출력 값을 도출하도록 하는 과정
    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3)+b3
    y = identify_function(a3)

    # 도출된 최종 출력값 (넘파이 배열) 리턴
    return y

# network에 신경망(가중치와 편향값) 딕셔너리 저장
network = init_network()
# 입력값을 넘파이 배열로 저장하여 x에 저장
x = np.array([1.0,0.5])
# network 신경망 딕셔너리, 입력값 배열 x를 입력하여 신경망 순방향 처리 후 결과 값을 y에 저장
y = forward(network, x)
# y(신경망 순방향 처리 결과 값 배열) 출력
print(y)

