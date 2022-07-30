# 오버피팅은 주로 가중치 매개변수의 값이 커서 발생하는 경우가 많음
# 따라서 가중치 값이 클수록 그에 맞게 큰 페널피를 부과하는 가중치 감소(weight decay) 기법 사용

# 손실함수의 값을 줄일 때 가중치의 제곱 노름(norm)을 손실함수에 더하면 가중치가 커지는 것을 억제 가능
# 가중치가 W일때 L2 노름에 따른 가중치 감소는 1/2*λ*W**2이 되고 이 값을 손실함수에 더함
# λ는 정규화의 세기를 조절하는 하이퍼파라미터

# 가중치 감소는 모든 가중치 각각의 손실함수에 1/2*λ*W**2 값을 더함
# 가중치 기울기를 구하는 계산에서 오차역전파법에 따른 결과에 정규화 항을 미분한 λW를 더함

# L2 노름은 각 원소들의 제곱들을 더한 것
# L1 노름은 각 원소들의 절댓값을 더한 것
# L∞ 노름은 각 원소의 절댓값 중 가장 큰 것
# 일반적으로 L2 노름을 자주 쓴다고 함


# λ를 0.1로 설정하여 가중치 감소를 적용하여 MNIST 신경망으로 테스트

import sys

sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

# MNIST 데이터셋 불러오기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 발생시키기 위해 훈련 데이터를 300개로 제한
x_train = x_train[:300]
t_train = t_train[:300]

# 가중치 감소 하이퍼파라미터 λ를 0.1로 설정
weight_decay_lambda = 0.1

# 가중치 감소를 적용한 신경망 생성
network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10, weight_decay_lambda=weight_decay_lambda)
# 매개변수 최적화 기법을 학습률 0.01인 SGD(확률적 경사 하강법) 사용
optimizer = SGD(lr=0.01)

# 최대 에폭수 201
max_epochs = 201
# 학습 크기를 x_train 행렬의 0번째 차원 크기(300)으로 설정
train_size = x_train.shape[0]
# 배치 크기 100
batch_size = 100

# 훈련 데이터 손실값을 저장할 리스트 생성
train_loss_list = []
# 훈련 데이터 추론 정확도를 저장할 리스트 생성
train_acc_list = []
# 시험 데이터 추론 정확도를 저장할 리스트 생성
test_acc_list = []

# 에폭당 반복 수는 학습 크기를 배치 크기로 나눈 값(3)과 1 중에서 큰 값
iter_per_epoch = max(train_size / batch_size, 1)
# 에폭 횟수를 저장할 변수
epoch_cnt = 0

# 최대한 많이 반복
for i in range(1000000000):
    # 훈련 크기의 값들 중에서 배치 크기의 수만큼 랜덤 인덱스 지정
    batch_mask = np.random.choice(train_size, batch_size)
    # 지정된 인덱스로 훈련용 배치 데이터 저장
    x_batch = x_train[batch_mask]
    # 지정된 인덱스로 훈련용 배치 데이터 정답 레이블 저장
    t_batch = t_train[batch_mask]

    # 훈련 데이터와 정답 레이블을 이용하여 신경망의 각 매개변수 기울기 계산
    grads = network.gradient(x_batch, t_batch)
    # 신경망 매개변수를 기울기를 이용하여 갱신
    optimizer.update(network.params, grads)

    # 현재 반복 수가 에폭당 반복 수의 배수라면
    if i % iter_per_epoch == 0:
        # 훈련 데이터 추론 정확도 계산
        train_acc = network.accuracy(x_train, t_train)
        # 시험 데이터 추론 정확도 계산
        test_acc = network.accuracy(x_test, t_test)
        # 현재 훈련 데이터 추론 정확도를 리스트에 원소로 추가
        train_acc_list.append(train_acc)
        # 현재 시험 데이터 추론 정확도를 리스트에 원소로 추가
        test_acc_list.append(test_acc)

        # 현재 에폭 횟수와 훈련 데이터 추론 정확도, 시험 데이터 추론 정확도 출력
        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        # 에폭 횟수 1 추가
        epoch_cnt += 1
        # 에폭 횟수가 최대 에폭수보다 크다면
        if epoch_cnt >= max_epochs:
            # 반복 종료
            break


# 훈련 데이터 추론 정확도와 시험 데이터 추론 정확도를 비교하는 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


# 가중치 감소를 적용했을 때 훈련 데이터 추론 정확도와 시험 데이터 추론 정확도의 차이가 상대적으로 작아짐
# 이와 별개로 가중치 감소 적용시 훈련 데이터의 추론 정확도가 떨어지는 현상도 확인 가능