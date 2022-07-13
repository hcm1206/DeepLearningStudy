# 신경망이 훈련 데이터 뿐만이 아닌 다른 데이터셋에서도 의도한 대로 작동하는지 확인하는 작업이 필요
# 오버피팅 : 훈련받은 데이터만 제대로 인식하고 그 외의 데이터를 제대로 인식하지 못하는 것
# 범용 능력을 평가하기 위해 훈련 받은 데이터 외에 별개의 데이터를 이용하여 인식 평가

# 1 에폭별로 훈련 데이터와 시험 데이터에 대한 정확도 기록
# 에폭 : 학습에서 훈련 데이터를 모두 소진했을 때의 횟수 (ex 10000개의 데이터를 100개의 미니배치로 학습하면 100회의 학습 뒤 모든 훈련데이터 소진, 이 때 100회가 1에폭)

import numpy as np
import sys
sys.path.append("/Users/hyeon/Documents/Python/DeepLearningStudy")
from common.gradient import numerical_gradient
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
# 각 에폭당 현재 훈련 데이터 계산 정확도를 저장할 리스트 생성
train_acc_list = []
# 각 에폭당 현재 시험 데이터 계산 정확도를 저장할 리스트 생성
test_acc_list = []

# 1 에폭당 반복 수 계산 (훈련 데이터 숫자/배치 크기 숫자) 또는 1 중에서 큰 값
iter_per_epoch = max(train_size/batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch,t_batch)

    for key in ['W1', 'b1', 'W2', 'b2']:
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에폭만큼 계산했다면 아래 내용 실행
    if i % iter_per_epoch == 0:
        # 훈련 데이터의 추론값과 실제 정답값과 비교한 정확도를 계산하여 현재 추론 정확도를 train_acc에 저장
        train_acc = network.accuracy(x_train,t_train)
        # 시험 데이터의 추론값과 실제 정답값과 비교한 정확도를 계산하여 현재 추론 정확도를 test_acc에 저장
        test_acc = network.accuracy(x_test, t_test)
        # 훈련 데이터 추론 정확도를 리스트에 추가하여 저장
        train_acc_list.append(train_acc)
        # 시험 데이터 추론 정확도를 리스트에 추가하여 저장
        test_acc_list.append(test_acc)
        # 훈련 데이터 추론 정확도와 시험 데이터 추론 정확도 출력
        print("train acc, test acc : " + str(train_acc) + ", " + str(test_acc))