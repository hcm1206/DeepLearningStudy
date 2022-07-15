# 신경망 학습에서 오차역전파법은 기울기 산출 단계에서 사용
# 오차역전파법을 적용하여 2층 신경망을 새로 구현

import sys
# 경로 오류로 인하여 절대경로 지정
sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
# 시그모이드 함수(활성화 함수), 소프트맥스 함수(출력 함수), 교차 엔트로피 오차 함수(손실 함수) 불러오기
from common.functions import *
# 편미분 함수 불러오기
from common.gradient import numerical_gradient
from collections import OrderedDict

import numpy as np

# Affine(가중치과 편향 계산) 계층
class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self,x):
        self.x = x
        out = np.dot(x,self.W) + self.b
        return out

    def backward(self,dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

# ReLU(활성화 함수) 계층
class ReLU:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

# 소프트맥스와 손실 함수(출력층) 계층
class SoftmaxWithLoss:
    def __init__(self):
        # 손실 값 배열 저장할 변수 생성
        self.loss = None
        # 추론 값 배열 저장할 변수 생성
        self.y = None
        # 정답 레이블 배열 저장할 변수 생성
        self.t = None

    # 순전파 계산 (입력 값과 정답 값 배열을 입력받음)
    def forward(self,x,t):
        # 정답 값 배열을 변수로 저장(역전파 계산 때 사용하기 위해)
        self.t = t
        # 소프트맥스 함수에 입력값 배열(x)를 입력하여 나온 추론 값 배열을 self.y 변수에 저장
        self.y = softmax(x)
        # 교차 엔트로피 함수에 추론 값 배열(self.y)와 정답 레이블 배열(self.t)를 입력하여 정규화된 추론 결과(손실 값)를 변수에 저장
        self.loss = cross_entropy_error(self.y, self.t)
        # 추론 결과(손실 값) 리턴
        return self.loss

    # 역전파 계산 (결과값 미분(기본 1) 입력받음)
    def backward(self, dout=1):
        # 배치 크기 지정 (정답 값 배열의 행(데이터)의 수)
        batch_size = self.t.shape[0]
        # 입력값의 역전파(미분) 계산은 추론 값에서 정답 값을 뺀 값(배열)을 배치 크기로 나누어 진행
        dx = (self.y-self.t)/batch_size
        # 입력값의 미분값 리턴
        return dx



# 2층 신경망 클래스 구현
class TwoLayerNet:
    # 생성자에서 입력층 크기, 은닉층 크기, 출력층 크기, 가중치의 초기 가중값(기본 0.01)을 입력받음
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 매개변수를 저장할 딕셔너리 생성
        self.params = {}
        # 1층 가중치 매개변수로 입력층 크기 × 은닉층 크기 형상의 행렬의 각 원소에 0부터 1까지의 표준정규분포 난수와 가중치 초기 가중값을 곱한 값을 저장
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 1층 편향 매개변수로 모든 원소가 0인 은닉층 크기의 배열 저장
        self.params['b1'] = np.zeros(hidden_size)
        # 2층 가중치 매개변수로 은닉층 크기 × 출력층 크기 형상의 행렬의 각 원소에 0부터 1까지의 표준정규분포 난수와 가중치 초기 가중값을 곱한 값을 저장
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # 2층 편향 매개변수로 모든 원소가 0인 은닉층 크기의 배열 저장
        self.params['b2'] = np.zeros(output_size)


        # 신경망의 모든 계층들을 순서대로 저장하기 위해 layers 클래스 변수에 순서를 체크하는 딕셔너리 저장
        self.layers = OrderedDict()
        # 'Affine1' 계층으로 첫번째 가중치와 첫번째 편향 값을 입력한 Affine 클래스 저장
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        # 'Relu1' 계층으로 활성화 함수로 사용할 ReLU 클래스 저장
        self.layers['Relu1'] = ReLU()
        # 'Affine2' 계층으로 두번째 가중치와 첫번째 편향 값을 입력한 Affine 클래스 저장
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        # 최종 계층으로 소프트맥스 함수와 교차엔트로피오차(손실) 클래스 저장(추론값 정규화)
        self.lastLayer = SoftmaxWithLoss()

    # 입력값을 토대로 결과값을 추론하는 함수
    def predict(self,x):
        # 신경망의 모든 계층을 실행하며 반복
        for layer in self.layers.values():
            # x(입력값)을 해당 계층 순전파에 입력하여 나온 결과값을 다시 x에 저장
            x = layer.forward(x)
        # 모든 계층에서 순전파 계산이 끝난 x의 최종 결과값 리턴
        return x
    
    # 입력값과 정답 레이블을 입력받아 오차를 확인하는 손실 함수
    def loss(self,x,t):
        # y에 x를 입력하여 모든 계층을 순방향 계산 후 추론한 결과값 저장
        y = self.predict(x)
        # 최종 계층(소프트맥스 함수와 교차 엔트로피 오차 함수 계산)에 추론값과 정답 레이블 입력한 값(평균 오차) 리턴
        return self.lastLayer.forward(y,t)

    # 입력값과 정답 레이블을 입력받아 정확도를 계산하는 함수
    def accuracy(self,x,t):
        # y에 신경망의 추론 값 저장
        y = self.predict(x)
        # y에 추론 값 배열 중에서 가장 크기가 큰 원소의 인덱스 저장(해당 인덱스가 정답이라고 추론했다는 뜻)
        y = np.argmax(y, axis=1)
        # 정답 레이블의 배열이 1차원이 아니라면(배치 형식이라면) 정답 레이블의 1번째 차원(개별 데이터의 정답 원-핫-인코딩)에서 가장 큰 원소의 인덱스를 t에 저장
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        # 추론값 배열(y)과 정답 레이블 배열(t)가 같은 배열의 합을 입력값 배열의 0차원의 수(입력되는 데이터 수)로 나누어 정확도 계산
        accuracy = np.sum(y == t) / float(x.shape[0])
        # 계산된 정확도 리턴
        return accuracy

    # 입력값과 정답 레이블을 입력받아 수치 미분을 통해 기울기를 구하는 함수
    def numerical_gradient(self, x, t):
        # 입력값과 정답 레이블을 손실 함수(교차 엔트로피 함수)를 통해 평균 손실값을 (람다함수로) 계산하여 loss_W에 저장
        loss_W = lambda W: self.loss(x, t)

        # 기울기를 저장할 딕셔너리 생성
        grads = {}
        # 첫번째 층 가중치 기울기를 총 손실 함수에서 현재 첫번째 층 가중치에 대해 수치 미분하여 구함
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        # 첫번째 층 편향 기울기를 총 손실 함수에서 현재 첫번째 층 편향에 대해 수치 미분하여 구함
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        # 두번쨰 층 가중치 기울기를 총 손실 함수에서 현재 두번째 층 가중치에 대해 수치 미분하여 구함
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        # 두번째 층 편향 기울기를 총 손실 함수에서 현재 두번째 층 편향에 대해 수치 미분하여 구함
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        # 기울기(딕셔너리) 리턴
        return grads

    # 입력값과 정답 레이블을 입력받아 역전파를 통해 기울기를 구하는 함수
    def gradient(self,x,t):
        # 입력값과 정답 레이블을 입력하여 순전파 계산(결과는 따로 저장하지 않고 역전파 계산에 필요한 변수만 저장)
        self.loss(x,t)

        # 최초 역전파 입력값(출력값의 미분)을 1로 정의
        dout = 1
        # 최종 계층(소프트맥스 함수와 교차 엔트로피 함수)에 dout(1)을 입력하여 역전파 계산한 값을 다시 dout에 저장
        dout = self.lastLayer.backward(dout)

        # layers에 각 계층 딕셔너리의 값(클래스)들을 리스트로 저장
        layers = list(self.layers.values())
        # layers 리스트의 원소 순서를 반대로 하기(순전파 계산 순서로 정렬된 각 계층의 순서를 반대로 지정)
        layers.reverse()
        # layers(역전파 계산을 위해 반대 순서로 정렬된 계층들)에서 각 계층을 순서대로 진행하며 반복
        for layer in layers:
            # dout(현재 미분값)을 현재 계층의 역전파 계산을 한 결과값을 다시 dout에 저장
            dout = layer.backward(dout)
        
        # 기울기를 저장할 딕셔너리 생성
        grads = {}
        # 첫번째 층 가중치의 기울기로 'Affine1' 계층의 dW 변수(첫번째 층 가중치의 기울기) 저장
        grads['W1'] = self.layers['Affine1'].dW
        # 첫번쨰 층 편향의 기울기로 'Affine1' 계층의 b1 변수(첫번쨰 층 편향의 기울기) 저장
        grads['b1'] = self.layers['Affine1'].db
        # 두번쨰 층 가중치의 기울기로 'Affine2' 계층의 dW 변수(두번쨰 층 가중치의 기울기) 저장
        grads['W2'] = self.layers['Affine2'].dW
        # 두번쨰 층 편향의 기울기로 'Affine2' 계층의 b2 변수(두번쨰 층 편향의 기울기) 저장
        grads['b2'] = self.layers['Affine2'].db

        # 계산된 기울기가 저장된 grads 딕셔너리 리턴
        return grads
