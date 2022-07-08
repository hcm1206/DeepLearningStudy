# 미니배치 학습을 통해 신경망 학습 구현
# 미니배치 학습 : 훈련 데이터의 일부를 꺼내 미니배치로 묶어서 한번에 학습하는 것

import sys, os
sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
import numpy as np
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet

# MNIST 손글씨 데이터셋의 훈련 이미지, 시험 이미지와 각각에 대응하는 훈련 정답 레이블, 시험 정답 레이블 저장
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# 매 훈련 시마다의 평균 손실값을 저장할 리스트 생성
train_loss_list = []
# 훈련 반복 횟수
iters_num = 10000
# 훈련하는 데이터 크기를 훈련 데이터의 0차원 크기(60000개)로 지정
train_size = x_train.shape[0]
# 한 번의 학습에 사용되는 데이터 묶음(배치) 크기를 100으로 지정
batch_size = 100
# 학습률(매 학습마다의 매개변수 갱신율)을 0.1로 지정
learning_rate = 0.1

# network에 입력층 뉴런 784, 은닉층 뉴런 50개, 출력층 뉴런 10개의 신경망 저장
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 훈련 반복 횟수만큼 반복
for i in range(iters_num):
    # 훈련 데이터 크기 범위의 수에서 배치 크기만큼의 랜덤 수를 뽑아 batch_mask에 배열로 저장
    batch_mask = np.random.choice(train_size, batch_size)
    # 모든 훈련 데이터 중 batch_mask 배열의 난수 인덱스의 훈련 데이터를 꺼내 x_batch에 저장
    x_batch = x_train[batch_mask]
    # 모든 훈련 데이터 정답 레이블 중 batch_mask 배열의 난수 인덱스의 훈련 데이터를 꺼내 t_batch에 저장
    t_batch = t_train[batch_mask]

    # 신경망의 각 매개변수에서의 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)

    # 각 매개변수들을 key에 대입하여 차례로 반복
    for key in ('W1', 'b1', 'W2', 'b2'):
        # network 신경망 params 딕셔너리에 저장된 해당 매개변수에서 계산된 기울기와 학습률을 곱한 값을 빼서 저장
        network.params[key] -= learning_rate * grad[key]
    # 신경망의 훈련 데이터와 훈련 데이터 정답 레이블을 손실함수를 통해 비교한 평균 손실값을 loss에 저장
    loss = network.loss(x_batch, t_batch)
    # 구한 평균 손실값을 손실 값 리스트에 원소로 추가 
    train_loss_list.append(loss)


# train_loss_list(평균 손실값이 저장된 리스트)의 원소들을 그래프로 차례로 그려보면 뒤로 갈수록 값이 점점 작아짐
# 학습 횟수가 늘어나면서 값이 줄어든다는 것은 학습이 잘 진행되며 매개변수가 최적화된 값으로 조절되고 있다는 뜻