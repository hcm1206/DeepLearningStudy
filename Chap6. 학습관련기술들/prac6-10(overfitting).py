# 오버피팅이란 신경망이 훈련 데이터에만 지나치게 적응하여 훈련 데이터 외의 데이터를 정확하게 추론하지 못하는 상태를 의미
# 기계학습에서 오버피팅이 일어나면 범용성이 떨어지는 문제 발생 (새로운 데이터를 제대로 추론하지 못함)

# 오버피팅의 조건
# 1. 매개변수가 많고 표현력이 높은 신경망 모델
# 2. 적은 훈련 데이터


# 오버피팅이 일어나는 예시 코드 작성
# 7층 네트워크를 사용하는 복잡한 신경망 모델 상에서 300개의 MNIST 훈련 데이터만을 사용하여 오버피팅 발생시키기

import sys

sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

# MNIST 데이터 로드
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 학습 데이터 수를 300개로 줄임 (오버피팅을 일으키기 위함)
x_train = x_train[:300]
t_train = t_train[:300]

# 신경망 생성
network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10)
# 가중치 매개변수 최적화 기법 : 학습률 0.01의 SGD(확률적 경사 하강법) 사용
optimizer = SGD(lr=0.01)

# 최대 에폭수 201
max_epochs = 201
# 학습 크기 : 학습 데이터 행렬의 0차원 크기 (300)
train_size = x_train.shape[0]
# 배치 크기 100
batch_size = 100

# 훈련 데이터의 손실 값 저장할 리스트 생성
train_loss_list = []
# 훈련 데이터의 추론 정확도 저장할 리스트 생성
train_acc_list = []
# 시험 데이터의 추론 정확도 저장할 리스트 생성
test_acc_list = []

# 각 에폭 당 반복 수를 훈련 데이터 크기(300)를 배치 크기(100)로 나눈 값(3) 또는 1 중 하나로 설정
iter_per_epoch = max(train_size / batch_size, 1)
# 에폭 횟수를 카운트할 변수 생성
epoch_cnt = 0

# 10억번 반복
for i in range(1000000000):
    # 훈련용 배치 데이터로 사용할 랜덤 인덱스 생성
    batch_mask = np.random.choice(train_size, batch_size)
    # 훈련 데이터 중 지정된 인덱스의 데이터를 배치 데이터로 저장
    x_batch = x_train[batch_mask]
    # 훈련 데이터 정답 레이블 중 지정된 인덱스의 데이터를 배치 데이터 정답 레이블로 저장
    t_batch = t_train[batch_mask]

    # 신경망 매개변수의 기울기 계산
    grads = network.gradient(x_batch, t_batch)
    # 계산된 기울기에 따라 매개변수 갱신
    optimizer.update(network.params, grads)

    # 에폭당 반복횟수를 반복했다면
    if i % iter_per_epoch == 0:
        # 신경망의 훈련 데이터 추론 정확도 계산
        train_acc = network.accuracy(x_train, t_train)
        # 신경망의 시험 데이터 추론 정확도 계산
        test_acc = network.accuracy(x_test, t_test)
        # 현재 훈련 데이터 추론 정확도를 리스트에 저장
        train_acc_list.append(train_acc)
        # 현재 시험 데이터 추론 정확도를 리스트에 저장
        test_acc_list.append(test_acc)

        # 현재 에폭 횟수와 훈련 정확도, 추론 정확도 출력
        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        # 에폭 횟수 1 증가
        epoch_cnt += 1
        # 에폭 횟수가 최대 에폭 횟수(201)에 도달하면 반복 종료
        if epoch_cnt >= max_epochs:
            break


# 훈련 데이터용 마커를 원형으로, 시험 데이터용 마커를 사각형으로 지정
markers = {'train': 'o', 'test': 's'}
# x축을 0~최대 에폭수(201)까지 범위의 배열로 지정
x = np.arange(max_epochs)
# x축에 따른 훈련 데이터 정확도를 그래프로 표시
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
# x축에 따른 시험 데이터 정확도를 그래프로 표시
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


# 그래프에 따르면 훈련 데이터의 추론 정확도는 매우 정확하지만 시험 데이터의 추론 정확도는 정확성이 다소 떨어짐