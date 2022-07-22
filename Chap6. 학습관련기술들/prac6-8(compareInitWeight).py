# MNIST 데이터를 이용하여 각 가중치 초깃값이 학습에 어떻게 영향을 주는지 그래프로 확인


import sys

sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD


# MNIST 데이터 불러오기(훈련 데이터, 훈련 데이터 정답, 시험 데이터, 시험 데이터 정답)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 훈련량 : 60000 (훈련 데이터의 0차원 크기)
train_size = x_train.shape[0]
# 배치 크기 128
batch_size = 128
# 최대 반복 횟수 2000
max_iterations = 2000



# 각 가중치 초기값 딕셔너리 설정
weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
# 최적화 기법은 학습률 0.01의 SGD(확률적 경사 하강법)기법 사용
optimizer = SGD(lr=0.01)

# 가중치 초깃값 별 신경망을 저장하는 딕셔너리 생성
networks = {}
# 가중치 초깃값 별 손실 값을 저장하는 딕셔너리 생성
train_loss = {}
# 가중치 초깃값과 사용할 활성화 함수 별로 반복
for key, weight_type in weight_init_types.items():
    # 해당 가중치 초깃값 별로 다층 신경망 생성
    networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100], output_size=10, weight_init_std=weight_type)
    # 해당 가중치 초기값을 사용한 신경망의 손실 값을 저장할 리스트 생성
    train_loss[key] = []


# 최대 반복 횟수만큼 반복
for i in range(max_iterations):
    # 배치 지정
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 각 가중치 초깃값 별로 반복
    for key in weight_init_types.keys():
        # 신경망의 기울기 계산
        grads = networks[key].gradient(x_batch, t_batch)
        # SGD 기법으로 기울기값 갱신
        optimizer.update(networks[key].params, grads)
        # 현재 신경망의 손실값 저장
        loss = networks[key].loss(x_batch, t_batch)
        # 신경망의 손실 값을 해당하는 가중치 초깃값 담당 손실값 리스트에 저장
        train_loss[key].append(loss)
    # 반복 횟수 100번 간격으로 아래 내용 실행
    if i % 100 == 0:
        # 현재 반복 횟수 출력
        print("===========" + "iteration:" + str(i) + "===========")
        # 각 가중치 초기값 방식 별로 반복
        for key in weight_init_types.keys():
            # 현재 가중치 초기값을 사용하는 신경망의 손실값 계산
            loss = networks[key].loss(x_batch, t_batch)
            # 손실값 출력
            print(key + ":" + str(loss))


# 표준편차 0.01 정규분포 그래프 점은 원형, Xavier 초기값 그래프 점은 사각형, He 초기값 그래프 점은 마름모로 지정
markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
# x에 0부터 최대 반복 횟수(20000)까지의 배열 저장
x = np.arange(max_iterations)
# 각 가중치 초기값 방식별로 반복
for key in weight_init_types.keys():
    # 각 가중치 초기값 방식의 학습 반복 횟수(x)에 따른 손실 함수 값(train_loss[key])을 그래프로 표현
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 2.5)
plt.legend()
plt.show()



# x축은 학습 시행 횟수, y축은 해당 가중치 초깃값을 지정한 신경망 별 손실 값(0에 가까울수록 학습이 잘 되었다는 뜻)

# std = 0.01(단순히 표준정규분포에 표준편차 0.01로 분포한 가중치) 일 때는 학습이 거의 이루어지지 않음
# 순전파 때 0에 가까운 아주 작은 값만이 흘러 역전파 때의 기울기가 작으므로 가중치 갱신이 원활하지 않기 때문

# Xavier와 He 초깃값에서는 학습이 원활하게 이루어지고 있음을 확인 가능

# 이렇듯 가중치 초깃값을 어떻게 설정하느냐에 따라 결과가 크게 달라짐