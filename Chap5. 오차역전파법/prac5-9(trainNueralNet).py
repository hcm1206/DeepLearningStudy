import numpy as np
import sys
sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
from common.gradient import numerical_gradient
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet

# 훈련에 사용할 데이터(mnist 데이터 셋) 불러오기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# 2층 신경망 저장
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 반복 횟수 10000
iters_num = 10000
# 훈련 데이터 크기 : x_train의 0번째 차원의 크기
train_size = x_train.shape[0]
# 배치 크기 100
batch_size = 100
# 학습률(매 학습마다의 매개변수 갱신률) 0.1
learning_rate = 0.1

# 에폭당 학습에서 평균 손실값을 저장할 리스트
train_loss_list = []
# 에폭당 학습 데이터 추론 정확도를 저장할 리스트
train_acc_list = []
# 에폭당 시험 데이터 추론 정확도를 저장할 리스트
test_acc_list = []
# 1 에폭당 반복 수 계산(학습 크기를 배치 크기로 나눈 값과 1 중 큰 값으로 지정)
iter_per_epoch = max(train_size/batch_size,1)

# 지정한 반복 횟수(10000)만큼 반복
for i in range(iters_num):
    # 총 학습 데이터 크기(60000) 중 배치 크기(100) 만큼의 랜덤 인덱스 저장
    batch_mask = np.random.choice(train_size, batch_size)
    # 훈련 데이터에서 지정된 배치 인덱스 만큼의 데이터를 가져와 배치 훈련 데이터로 지정
    x_batch = x_train[batch_mask]
    # 정답 데이터에서 지정된 배치 인덱스 만큼의 데이터를 가저와 배치 정답 데이터로 지정
    t_batch = t_train[batch_mask]
    # 훈련 배치 데이터와 정답 배치 데이터의 기울기를 오차역전파법을 통해 구함
    grad = network.gradient(x_batch, t_batch)

    # 각 매개변수 별로 반복
    for key in ('W1', 'b1', 'W2', 'b2'):
        # 해당 매개변수에서 학습률과 해당 매개변수의 기울기를 곱한 값만큼 뺌
        network.params[key] -= learning_rate * grad[key]
    # 훈련 배치 데이터와 정답 배치 데이터를 비교하여 평균 오차를 구하여 저장
    loss = network.loss(x_batch, t_batch)
    # 현재 평균 오차를 리스트에 저장
    train_loss_list.append(loss)

    # 현재 반복 횟수가 에폭별 반복 횟수의 배수라면 아래 내용 실행(100번 반복마다 아래 내용 실행)
    if i % iter_per_epoch == 0:
        # 훈련 데이터 추론의 현재 정확도를 train_acc에 저장
        train_acc = network.accuracy(x_train, t_train)
        # 시험 데이터 추론의 현재 정확도를 test_acc에 저장
        test_acc = network.accuracy(x_test, t_test)
        # 현재 훈련 데이터 추론 정확도를 리스트에 원소로 저장
        train_acc_list.append(train_acc)
        # 현재 시험 데이터 추론 정확도를 리스트에 원소로 저장
        test_acc_list.append(test_acc)
        # 현재 훈련 데이터 추론 정확도와 현재 시험 데이터 추론 정확도 출력
        print(train_acc, test_acc)