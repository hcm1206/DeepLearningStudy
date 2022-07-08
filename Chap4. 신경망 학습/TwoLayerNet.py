# 2층 신경망 클래스 구현

import sys, os
# 경로 오류로 인하여 절대경로 지정
sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
# 시그모이드 함수(활성화 함수), 소프트맥스 함수(출력 함수), 교차 엔트로피 오차 함수(손실 함수) 불러오기
from common.functions import *
# 편미분 함수 불러오기
import numpy as np

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

def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        return grad


class TwoLayerNet:
    # 입력층 뉴런 수, 은닉층 뉴런 수, 출력층 뉴런 수, 가중치 초기 기준값(기본 0.01)을 인수로 받는 클래스 생성자
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 매개변수가 저장될 딕셔너리 self.params 생성
        self.params = {}
        # 첫번째 층 가중치 생성 (입력층 뉴런 수 × 은닉층 뉴런 수 형상의 행렬에 0~1 사이 랜덤 정규분포 수 × 가중치 초기 기준값 저장)
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 첫번쨰 층 편향 생성 (은닉층의 뉴런 수 크기의 배열을 생성하여 모두 0으로 채움)
        self.params['b1'] = np.zeros(hidden_size)
        # 두번째 층 가중치 생성 (은닉층 뉴런 수 × 출력층 뉴런 수 형상의 행렬에 0~1 사이 랜덤 정규분포 수 × 가중치 초기 기준값 저장)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # 두번째 층 편향 생성 (출력층의 뉴런 수 크기의 배열을 생성하여 모두 0으로 채움)
        self.params['b2'] = np.zeros(output_size)

    # 입력값(x 배열)을 입력받아 예측값 출력하는 추론 함수
    def predict(self,x):
        # W1, W2에 params 딕셔너리에 배열로 저장된 가중치를 각각 불러와 저장
        W1,W2 = self.params['W1'],self.params['W2']
        # b1, b2에 params 딕셔너리에 배열로 저장된 편향을 각각 불러와 저장
        b1,b2 = self.params['b1'],self.params['b2']

        # 입력값(x 배열)과 첫번째 층 가중치(W1 배열)을 행렬곱하고 첫번째 층 편향(b1 배열)을 더한 첫번째 층 계산 결과를 a1에 저장
        a1 = np.dot(x,W1) + b1
        # 첫번째 층 계산 결과(a1 배열)를 활성화 함수(시그모이드 함수)에 입력하여 나온 활성화 결과 값을 z1(은닉층 배열)에 저장
        z1 = sigmoid(a1)
        # 은닉값(z1 배열)과 두번째 층 가중치(W2 배열)을 행렬곱하고 두번째 층 편향(b2 배열)을 더한 두번째 층 계산 결과를 a2에 저장
        a2 = np.dot(z1,W2) + b2
        # 두번째 층 계산 결과(a2 배열)를 출력 함수(소프트맥스 함수)에 입력하여 나온 추론 결과 값을 y(출력층 배열)에 저장
        y = softmax(a2)

        # y(출력층 배열) 리턴
        return y

    # 입력 레이블(x 배열)과 정답 레이블(t 배열)을 입력받는 손실 함수
    def loss(self,x,t):
        # 입력값 x 배열의 결과 값 추론 배열을 y에 저장
        y = self.predict(x)
        # 교차 엔트로피 오차 함수에 추론한 결과 값 배열과 실제 정답 배열을 입력하여 비교한 평균 오차값 리턴
        return cross_entropy_error(y,t)
    
    # 입력 레이블(x 배열)과 정답 레이블(t 배열)을 입력받아 정확도 측정하는 함수
    def accuracy(self,x,t):
        # y에 입력 레이블(x 배열)의 결과 값 추론 배열 저장
        y = self.predict(x)
        # y에 y 배열의 1차원에서 가장 큰 값의 인덱스 저장
        y = np.argmax(y,axis=1)
        # t에 t 배열의 1차원에서 가장 큰 값의 인덱스 저장
        t = np.argmax(t,axis=1)

        # y와 t가 같은 배열의 합을 입력값 배열(x)의 0차원의 수(입력되는 데이터 수)로 나누어 정확도 계산
        accuracy = np.sum(y == t) / float(x.shape[0])
        # 계산된 정확도 리턴
        return accuracy

    # 입력값 배열(x)와 정답 배열(t)를 입력받아 편미분(기울기) 계산하는 함수
    def numerical_gradient(self,x,t):
        # W(그냥 의미없는 가중치)를 입력받아 추론 값(x)과 정답 값(t)의 평균 오차값을 출력하는 loss_W 람다 함수 정의
        loss_W = lambda W: self.loss(x,t)


        # 각 매개변수의 편미분(기울기) 값을 저장할 grads 딕셔너리 생성
        grads = {}
        # 손실함수에서 첫번째 가중치 값에서의 편미분(기울기) 값 저장
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        # 손실함수에서 첫번째 편향 값에서의 편미분(기울기) 값 저장
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        # 손실함수에서 두번째 가중치 값에서의 편미분(기울기) 값 저장
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        # 손실함수에서 두번째 편향 값에서의 편미분(기울기) 값 저장
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])


        # 편미분(기울기) 값이 저장된 딕셔너리 리턴
        return grads


# __init__ 생성자를 통해 클래스 초기화
# MNIST 손글씨 숫자 인식에서는 크기가 28×28인 784개 입력 이미지 데이터인 입력값, 0~9인 10개 출력값이므로 이에 맞게 변수 지정, 은닉층의 개수는 적당한 값 지정
# 생성자 메소드에서는 가중치 매개변수도 초기화, 가중치 매개변수는 정규분포를 따루는 난수로, 편향은 0으로 초기화됨
# 기울기는 numerical_gradient() 메소드처럼 수치 미분을 사용할 수도 있지만 오차역전파법을 통해 더 빠르게 기울기를 구할 수도 있음. 다음 장에서 확인